// Document Intelligence Agent — vanilla JS front-end. Talks to the FastAPI app
// on the same origin: /documents/* for ingest, /agent for the multi-agent turn.

const API = "";                 // same origin
const state = {
  messages: [],                 // conversation sent to /agent: {role, content}
  docs: {},                     // doc_id -> {name, pages, chunks}
  mode: "hybrid",
  scope: "",                    // "" = all docs, else a doc_id
  busy: false,
};

const $ = (id) => document.getElementById(id);
const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

async function api(path, opts = {}) {
  const res = await fetch(API + path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
  return body;
}

// ---- document ingest ------------------------------------------------------
async function ingest(file) {
  if (!file || file.type !== "application/pdf") { alert("Please choose a PDF."); return; }
  const drop = $("drop");
  drop.textContent = `Uploading ${file.name}…`;
  try {
    const fd = new FormData();
    fd.append("file", file);
    const up = await api("/documents/upload", { method: "POST", body: fd });
    await api(`/documents/${up.doc_id}/process`, { method: "POST" });
    const idx = await api(`/documents/${up.doc_id}/index`, { method: "POST" });
    state.docs[up.doc_id] = { name: up.original_filename, pages: idx.total_pages, chunks: idx.total_chunks };
    renderDocs();
  } catch (e) {
    alert("Upload failed: " + e.message);
  } finally {
    drop.innerHTML = '<input type="file" id="file" accept="application/pdf">Drop a PDF here or click to upload';
    wireDrop();
  }
}

function renderDocs() {
  const ids = Object.keys(state.docs);
  const list = $("docs");
  const scope = $("scope");
  if (!ids.length) { list.innerHTML = '<div class="empty">No documents yet.</div>'; return; }
  list.innerHTML = ids.map((id) => {
    const d = state.docs[id];
    return `<div class="doc">${esc(d.name)}<div class="meta">${d.pages} pages · ${d.chunks} chunks</div></div>`;
  }).join("");
  scope.innerHTML = '<option value="">All documents</option>' +
    ids.map((id) => `<option value="${id}">${esc(state.docs[id].name)}</option>`).join("");
  scope.value = state.scope;
}

// ---- chat -----------------------------------------------------------------
function bubble(role, inner) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role === "user" ? "user" : "bot"}`;
  wrap.innerHTML = inner;
  $("messages").appendChild(wrap);
  $("messages").scrollTop = $("messages").scrollHeight;
  return wrap;
}

function renderResult(r) {
  const status = r.status || "answered";
  const label = { answered: "answered", summary: "summary", insufficient_evidence: "no answer in docs",
                  smalltalk: "assistant" }[status] || status;
  const pillClass = status === "answered" && r.intent === "summarize" ? "summary" : status;
  let html = `<div class="bubble"><span class="pill ${pillClass}">${label}</span>${esc(r.answer || "")}`;
  if (r.sources && r.sources.length) {
    html += `<div class="sources">` + r.sources.map((s) =>
      `<span class="src">${esc(docName(s.doc_id))} · p.${s.page_number}</span>`).join("") + `</div>`;
  }
  if (r.trace && r.trace.length) {
    html += `<details class="trace"><summary>agent steps</summary><pre>${esc(JSON.stringify(r.trace, null, 2))}</pre></details>`;
  }
  html += `</div>`;
  return html;
}

function docName(id) { return state.docs[id] ? state.docs[id].name : id; }

async function send() {
  const input = $("input");
  const text = input.value.trim();
  if (!text || state.busy) return;
  input.value = ""; input.style.height = "auto";
  state.busy = true; $("send").disabled = true;

  bubble("user", `<div class="bubble">${esc(text)}</div>`);
  state.messages.push({ role: "user", content: text });
  const typing = bubble("bot", `<div class="bubble typing">thinking…</div>`);

  try {
    const r = await api("/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: state.messages, mode: state.mode, doc_id: state.scope || null }),
    });
    typing.outerHTML = `<div class="msg bot">${renderResult(r)}</div>`;
    state.messages.push({ role: "assistant", content: r.answer || "" });
  } catch (e) {
    const hint = /unavailable|503/i.test(e.message)
      ? "The local model isn't reachable. Start it (llama.cpp serve.sh, or Ollama) and try again."
      : e.message;
    typing.outerHTML = `<div class="msg bot"><div class="bubble"><span class="pill error">error</span>${esc(hint)}</div></div>`;
  } finally {
    state.busy = false; $("send").disabled = false;
    $("messages").scrollTop = $("messages").scrollHeight;
  }
}

// ---- wiring ---------------------------------------------------------------
function wireDrop() {
  const drop = $("drop");
  const file = $("file");
  file.addEventListener("change", () => file.files[0] && ingest(file.files[0]));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("over");
    if (e.dataTransfer.files[0]) ingest(e.dataTransfer.files[0]);
  });
}

document.querySelectorAll("#mode button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll("#mode button").forEach((x) => x.classList.remove("on"));
    b.classList.add("on"); state.mode = b.dataset.mode;
  }));
$("scope").addEventListener("change", (e) => { state.scope = e.target.value; });
$("send").addEventListener("click", send);
$("input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
$("input").addEventListener("input", (e) => { e.target.style.height = "auto"; e.target.style.height = e.target.scrollHeight + "px"; });
wireDrop();
