"""Image / figure extraction with a LOCAL vision model.

When the user asks to extract or describe images on a page, we render that page to
a PNG and send it to a local VLM (served by Ollama, e.g. a qwen2-vl or
granite-docling) over the OpenAI-compatible API. Nothing leaves the machine.

Disabled unless VISION_MODEL is set — then it returns a clear, honest message so
the agent can say "image extraction isn't configured" instead of pretending.
"""
from __future__ import annotations

import base64

import httpx

from app import config

_DEFAULT_PROMPT = (
    "Extract and transcribe any images, figures, charts, diagrams, or photos on this "
    "page. Describe what each shows and transcribe any text inside it. If the page has "
    "no images or figures, say so plainly."
)


def extract_page_images(doc_id: str, page_number: int, instruction: str | None = None) -> dict:
    if not config.VISION_MODEL:
        return {"error": "Image extraction isn't configured. Set VISION_MODEL to a local "
                         "vision model (e.g. a VLM in Ollama) to enable it."}

    pdf = config.UPLOAD_DIR / f"{doc_id}.pdf"
    if not pdf.exists():
        return {"error": f"Document '{doc_id}' not found."}

    try:
        import pymupdf
        doc = pymupdf.open(str(pdf))
        if page_number < 1 or page_number > doc.page_count:
            doc.close()
            return {"error": f"Page {page_number} is out of range (1–{doc.page_count})."}
        png = doc[page_number - 1].get_pixmap(dpi=150).tobytes("png")
        doc.close()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not render page {page_number}: {exc}"}

    b64 = base64.b64encode(png).decode()
    payload = {
        "model": config.VISION_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": instruction or _DEFAULT_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        "temperature": 0.0, "max_tokens": 700, "stream": False,
    }
    url = config.VISION_BASE_URL.rstrip("/") + "/chat/completions"
    try:
        r = httpx.post(url, json=payload,
                       headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
                       timeout=config.LLM_TIMEOUT)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Local vision model call failed: {exc}"}

    return {"page": page_number, "extraction": content}
