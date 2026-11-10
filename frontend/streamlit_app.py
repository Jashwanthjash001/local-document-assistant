"""
Local AI Document Assistant — Streamlit frontend (Stage 6).

A modern, dark glassmorphism UI over the FastAPI backend:
  upload -> index -> ask -> grounded answer with page citations.

The LLM server can stay OFF: "Demo mode" (default) answers from retrieved passages
without calling the model, so the whole UX is demonstrable offline. Toggle it off
to hit the real /chat (requires edge-ai-client/serve.sh running).

Run:
  # terminal 1 — backend
  .venv/bin/uvicorn app.main:app --port 8000
  # terminal 2 — frontend
  .venv/bin/streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import io
import os

import httpx
import pandas as pd
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Local Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling — modern dark glassmorphism
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root{
  --accent1:#8b5cf6; --accent2:#22d3ee; --accent3:#f472b6;
  --glass:rgba(255,255,255,0.04); --glass-border:rgba(255,255,255,0.09);
  --ink:#e6e9f2; --muted:#9aa3bd;
}
html, body, [class*="css"]  { font-family:'Inter',sans-serif; }
#MainMenu, footer, header {visibility:hidden;}
.stApp{
  background:
    radial-gradient(1100px 600px at 12% -10%, rgba(139,92,246,0.18), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(34,211,238,0.12), transparent 55%),
    linear-gradient(180deg,#0b1020 0%, #0a0e1a 100%);
}
.block-container{padding-top:1.6rem; max-width:1200px;}

/* Hero header */
.hero{ margin-bottom:.4rem; }
.hero h1{
  font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2.5rem;
  margin:0; line-height:1.05;
  background:linear-gradient(90deg,#c4b5fd,#8b5cf6 40%,#22d3ee);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.hero p{ color:var(--muted); margin:.35rem 0 0; font-size:1.02rem; }
.badge-row{ display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.8rem;}
.badge{
  font-size:.74rem; font-weight:600; letter-spacing:.02em;
  padding:.28rem .7rem; border-radius:999px;
  background:var(--glass); border:1px solid var(--glass-border); color:var(--ink);
}
.badge .dot{ color:#34d399; }

/* Glass cards */
.glass{
  background:var(--glass); border:1px solid var(--glass-border);
  border-radius:18px; padding:1.1rem 1.2rem; backdrop-filter:blur(14px);
  box-shadow:0 10px 30px rgba(0,0,0,0.25);
}

/* Chat bubbles */
.bubble-row{ display:flex; margin:.5rem 0; }
.bubble-row.user{ justify-content:flex-end; }
.bubble{
  max-width:78%; padding:.85rem 1.05rem; border-radius:16px; line-height:1.5;
  border:1px solid var(--glass-border);
}
.bubble.user{
  background:linear-gradient(135deg,rgba(139,92,246,0.9),rgba(99,102,241,0.85));
  color:white; border-bottom-right-radius:5px;
}
.bubble.assistant{
  background:var(--glass); color:var(--ink); border-bottom-left-radius:5px;
  backdrop-filter:blur(10px);
}
.bubble.assistant.abstain{ border-color:rgba(251,191,36,0.4); background:rgba(251,191,36,0.07); }
.bubble.assistant.error{ border-color:rgba(248,113,113,0.45); background:rgba(248,113,113,0.08); }

/* Source pills */
.pills{ display:flex; gap:.4rem; flex-wrap:wrap; margin:.55rem 0 0; }
.pill{
  font-size:.74rem; font-weight:600; color:#d9d4ff;
  padding:.26rem .62rem; border-radius:999px;
  background:rgba(139,92,246,0.14); border:1px solid rgba(139,92,246,0.4);
}
.pill .p{ color:var(--accent2); }

.statuschip{ font-size:.72rem; font-weight:700; letter-spacing:.03em; text-transform:uppercase; }
.statuschip.answered{ color:#34d399; }
.statuschip.insufficient_evidence{ color:#fbbf24; }
.statuschip.error{ color:#f87171; }

/* Buttons */
.stButton>button, .stDownloadButton>button{
  background:linear-gradient(135deg,var(--accent1),#6366f1); color:white; border:0;
  border-radius:12px; font-weight:600; padding:.5rem 1rem; transition:transform .08s ease, filter .15s ease;
}
.stButton>button:hover{ filter:brightness(1.08); transform:translateY(-1px); }

/* Sidebar */
section[data-testid="stSidebar"]{ background:rgba(10,14,26,0.85); border-right:1px solid var(--glass-border);}
.doc-item{ font-size:.85rem; padding:.5rem .6rem; border-radius:10px; background:var(--glass);
  border:1px solid var(--glass-border); margin-bottom:.45rem; }
.doc-item b{ color:#c4b5fd; }
.small{ color:var(--muted); font-size:.8rem; }
hr{ border-color:var(--glass-border); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Backend client
# --------------------------------------------------------------------------- #
def _url(path: str) -> str:
    return f"{BACKEND_URL.rstrip('/')}{path}"


def backend_online() -> bool:
    try:
        httpx.get(_url("/health"), timeout=3)
        return True
    except Exception:
        return False


def api_upload(filename: str, data: bytes) -> dict:
    r = httpx.post(_url("/documents/upload"),
                   files={"file": (filename, data, "application/pdf")}, timeout=30)
    r.raise_for_status()
    return r.json()


def api_index(doc_id: str) -> dict:
    r = httpx.post(_url(f"/documents/{doc_id}/index"), timeout=120)
    r.raise_for_status()
    return r.json()


def api_search(query: str, top_k: int, doc_id: str | None) -> list[dict]:
    r = httpx.post(_url("/search"),
                   json={"query": query, "top_k": top_k, "doc_id": doc_id}, timeout=30)
    r.raise_for_status()
    return r.json()["results"]


def api_chat(query: str, top_k: int, doc_id: str | None) -> dict:
    r = httpx.post(_url("/chat"),
                   json={"query": query, "top_k": top_k, "doc_id": doc_id}, timeout=120)
    if r.status_code == 503:
        return {"status": "error", "answer": "The local LLM is not running. "
                "Start edge-ai-client/serve.sh, or use Demo mode.", "sources": []}
    r.raise_for_status()
    return r.json()


def build_sample_pdf() -> bytes:
    import pymupdf
    pages = [
        "SERVICE AGREEMENT\nThis agreement is between Acme Corp and Contoso Ltd, effective January 1, 2025.",
        "TERMINATION\nThe termination notice period is 30 days. Either party may terminate with written notice.",
        "PAYMENT\nPayment is due within 14 days of invoice. Late payments incur a 2 percent monthly fee.",
        "CONFIDENTIALITY\nConfidential information must not be disclosed to third parties for a period of 5 years.",
    ]
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 90), text, fontsize=12)
    buf = doc.tobytes()
    doc.close()
    return buf


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
if "docs" not in st.session_state:
    st.session_state.docs = {}       # doc_id -> {filename, pages, chunks}
if "messages" not in st.session_state:
    st.session_state.messages = []   # {role, content, status, sources, passages}


def filename_for(doc_id: str) -> str:
    return st.session_state.docs.get(doc_id, {}).get("filename", "document")


def ingest(filename: str, data: bytes):
    up = api_upload(filename, data)
    doc_id = up["doc_id"]
    stats = api_index(doc_id)
    st.session_state.docs[doc_id] = {
        "filename": filename,
        "pages": stats["total_pages"],
        "chunks": stats["total_chunks"],
    }
    return doc_id, stats


def answer_query(query: str, top_k: int, threshold: float, doc_id: str | None, demo: bool) -> dict:
    """Return {answer, status, sources, passages}. Demo mode never calls the LLM."""
    passages = api_search(query, top_k, doc_id)           # CPU-only, no LLM
    relevant = [p for p in passages if p["distance"] <= threshold]

    if not relevant:
        return {"status": "insufficient_evidence",
                "answer": "I couldn't find this information in the uploaded documents.",
                "sources": [], "passages": passages}

    if demo:
        top = relevant[0]
        answer = ("**Demo answer (LLM offline).** Based on the uploaded documents: "
                  f"“{top['text']}” [Source 1]")
        sources = [{"source_id": f"Source {i+1}", **p} for i, p in enumerate(relevant)]
        return {"status": "answered", "answer": answer, "sources": sources, "passages": relevant}

    # Live mode: real grounded generation.
    result = api_chat(query, top_k, doc_id)
    result["passages"] = relevant
    return result


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 📄 Documents")
    online = backend_online()
    st.markdown(
        f"<div class='small'>Backend: "
        f"{'🟢 connected' if online else '🔴 offline — start uvicorn on :8000'}</div>",
        unsafe_allow_html=True,
    )

    demo_mode = st.toggle("Demo mode (LLM offline)", value=True,
                          help="On: answers from retrieved passages, no GPU. Off: calls the real local Llama.")
    top_k = st.slider("Passages to retrieve (top-k)", 1, 10, 5)
    threshold = st.slider("Relevance threshold (cosine distance)", 0.1, 1.5, 0.6, 0.05,
                          help="Lower = stricter. Below this = relevant. Nothing below = abstain.")

    st.markdown("---")
    up_file = st.file_uploader("Upload a PDF", type=["pdf"], disabled=not online)
    c1, c2 = st.columns(2)
    if c1.button("Index PDF", disabled=not (online and up_file)):
        with st.spinner("Extracting, chunking, embedding…"):
            try:
                _, stats = ingest(up_file.name, up_file.getvalue())
                st.success(f"Indexed {stats['total_chunks']} chunks")
            except Exception as e:
                st.error(f"Failed: {e}")
    if c2.button("Load sample", disabled=not online, help="Index a demo contract"):
        with st.spinner("Building & indexing sample contract…"):
            try:
                _, stats = ingest("sample_contract.pdf", build_sample_pdf())
                st.success(f"Sample indexed ({stats['total_chunks']} chunks)")
            except Exception as e:
                st.error(f"Failed: {e}")

    if st.session_state.docs:
        st.markdown("<div class='small'>Indexed documents</div>", unsafe_allow_html=True)
        for d in st.session_state.docs.values():
            st.markdown(
                f"<div class='doc-item'><b>{d['filename']}</b><br>"
                f"<span class='small'>{d['pages']} pages · {d['chunks']} chunks</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<div class='small'>No documents yet — upload one or load the sample.</div>",
                    unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
      <h1>Local Document Assistant</h1>
      <p>Ask questions about your PDFs — answered from the documents, on your machine, with page citations.</p>
      <div class="badge-row">
        <span class="badge"><span class="dot">●</span> 100% local</span>
        <span class="badge">Llama 3.1 8B</span>
        <span class="badge">ChromaDB · MiniLM embeddings</span>
        <span class="badge">Grounded + cited</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

tab_chat, tab_eval, tab_about = st.tabs(["  💬 Chat  ", "  📊 Evaluation  ", "  ⓘ About  "])


# --------------------------------------------------------------------------- #
# Chat tab
# --------------------------------------------------------------------------- #
def render_message(msg: dict):
    role = msg["role"]
    if role == "user":
        st.markdown(f"<div class='bubble-row user'><div class='bubble user'>{msg['content']}</div></div>",
                    unsafe_allow_html=True)
        return

    status = msg.get("status", "answered")
    cls = "assistant"
    if status == "insufficient_evidence":
        cls += " abstain"
    elif status == "error":
        cls += " error"

    pills = ""
    for s in msg.get("sources", []):
        fname = filename_for(s.get("doc_id", ""))
        pills += (f"<span class='pill'>{s['source_id']} · {fname} · "
                  f"<span class='p'>p.{s.get('page_number','?')}</span></span>")
    pills_html = f"<div class='pills'>{pills}</div>" if pills else ""

    st.markdown(
        f"<div class='bubble-row'><div class='bubble {cls}'>"
        f"<span class='statuschip {status}'>{status.replace('_',' ')}</span><br>"
        f"{msg['content']}{pills_html}</div></div>",
        unsafe_allow_html=True,
    )

    passages = msg.get("passages", [])
    if passages:
        with st.expander(f"🔎 Retrieved passages ({len(passages)})"):
            for i, p in enumerate(passages, 1):
                st.markdown(
                    f"**Source {i}** · {filename_for(p.get('doc_id',''))} · "
                    f"page {p.get('page_number','?')} · distance `{p['distance']:.3f}`")
                st.markdown(f"<div class='small'>{p['text']}</div>", unsafe_allow_html=True)
                st.markdown("---")


with tab_chat:
    if not st.session_state.messages:
        st.markdown(
            "<div class='glass'>👋 <b>Start by loading a document</b> (sidebar → "
            "<i>Load sample</i> or upload a PDF), then ask a question below — "
            "try <i>“How much notice is required to terminate?”</i></div>",
            unsafe_allow_html=True,
        )
    for msg in st.session_state.messages:
        render_message(msg)

    prompt = st.chat_input("Ask a question about your documents…")
    if prompt:
        if not st.session_state.docs:
            st.toast("Load or upload a document first.", icon="⚠️")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            try:
                res = answer_query(prompt, top_k, threshold, None, demo=demo_mode)
            except Exception as e:
                res = {"status": "error", "answer": f"Request failed: {e}", "sources": [], "passages": []}
            st.session_state.messages.append({
                "role": "assistant",
                "content": res["answer"],
                "status": res["status"],
                "sources": res.get("sources", []),
                "passages": res.get("passages", []),
            })
            st.rerun()


# --------------------------------------------------------------------------- #
# Evaluation tab
# --------------------------------------------------------------------------- #
EVAL_SET = [
    {"question": "How much notice is required to terminate?", "expect": "30 days"},
    {"question": "When are payments due?", "expect": "14 days"},
    {"question": "How long does confidentiality last?", "expect": "5 years"},
    {"question": "Who are the parties to the agreement?", "expect": "acme"},
    {"question": "What is the governing law of the contract?", "expect": "__ABSTAIN__"},
]

with tab_eval:
    st.markdown(
        "<div class='glass'>A tiny gold-set over the <b>sample contract</b>. Each row checks "
        "retrieval + the abstention gate. The last question has no answer in the document — "
        "the assistant should <b>abstain</b>.</div>", unsafe_allow_html=True)
    st.write("")
    if st.button("▶ Run evaluation", disabled=not st.session_state.docs):
        rows = []
        for item in EVAL_SET:
            try:
                passages = api_search(item["question"], top_k, None)
            except Exception as e:
                rows.append({"Question": item["question"], "Result": f"error: {e}",
                             "Top distance": None, "Pass": False})
                continue
            relevant = [p for p in passages if p["distance"] <= threshold]
            top_dist = round(passages[0]["distance"], 3) if passages else None
            if item["expect"] == "__ABSTAIN__":
                passed = len(relevant) == 0
                result = "abstained ✓" if passed else "answered ✗ (should abstain)"
            else:
                joined = " ".join(p["text"].lower() for p in relevant)
                passed = item["expect"].lower() in joined
                result = "found ✓" if passed else "missing ✗"
            rows.append({"Question": item["question"], "Result": result,
                         "Top distance": top_dist, "Pass": passed})

        df = pd.DataFrame(rows)
        passed_n = int(df["Pass"].sum())
        m1, m2 = st.columns(2)
        m1.metric("Passed", f"{passed_n}/{len(df)}")
        m2.metric("Pass rate", f"{100*passed_n/len(df):.0f}%")
        st.dataframe(df.drop(columns=["Pass"]), use_container_width=True, hide_index=True)
        chart = df.dropna(subset=["Top distance"]).set_index("Question")["Top distance"]
        if not chart.empty:
            st.markdown("<div class='small'>Top retrieved distance per question "
                        "(lower = closer match)</div>", unsafe_allow_html=True)
            st.bar_chart(chart, height=240)
    elif not st.session_state.docs:
        st.info("Load the sample document in the sidebar first.")


# --------------------------------------------------------------------------- #
# About tab
# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown(
        """
        <div class='glass'>
        <h3 style='margin-top:0'>How it works</h3>
        <p class='small'>Upload → text extracted per page (PyMuPDF) → chunked (~200 tokens, 30 overlap)
        → embedded on CPU (all-MiniLM-L6-v2, 384-dim) → stored in ChromaDB. A question is embedded
        the same way, the nearest chunks are retrieved by cosine distance, a relevance gate decides
        whether there's enough evidence, and a grounded prompt is sent to a local Llama 3.1 8B — which
        answers <b>only from the passages</b> and cites the source page, or abstains.</p>
        <p class='small'><b>Nothing leaves the machine.</b> Demo mode here answers from retrieved
        passages so the UI runs without the GPU; switch it off to use the real local model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
