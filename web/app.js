// Document Intelligence Agent — vanilla JS front-end. Same-origin calls to the
// FastAPI app: /documents/* to ingest, /agent for the multi-agent turn.

const API = "";
const state = { messages: [], docs: {}, mode: "hybrid", scope: "", busy: false };

const $ = (id) => document.getElementById(id);
const esc = (s) => (s || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const scrollDown = () => { $("messages").scrollTop = $("messages").scrollHeight; };
const hideWelcome = () => { const w = $("welcome"); if (w) w.remove(); };
const docName = (id) => (state.docs[id] ? state.docs[id].name : id);

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
  const reset = () => { drop.innerHTML = '<input type="file" id="file" accept="application/pdf"><b>Drop a PDF</b>or click to upload'; wireDrop(); };
  drop.innerHTML = `Uploading ${esc(file.name)}…`;
  try {
    const fd = new FormData(); fd.append("file", file);
    const up = await api("/documents/upload", { method: "POST", body: fd });
    await api(`/documents/${up.doc_id}/process`, { method: "POST" });
    const idx = await api(`/documents/${up.doc_id}/index`, { method: "POST" });
    state.docs[up.doc_id] = { name: up.original_filename, pages: idx.total_pages, chunks: idx.total_chunks };
    renderDocs();
  } catch (e) { alert("Upload failed: " + e.message); }
  finally { reset(); }
}

function renderDocs() {
  const ids = Object.keys(state.docs);
  const list = $("docs"), scope = $("scope");
  if (!ids.length) { list.innerHTML = '<div class="empty">No documents yet.</div>'; return; }
  list.innerHTML = ids.map((id) => {
    const d = state.docs[id];
    return `<div class="doc"><span class="dot"></span><div class="nm">${esc(d.name)}` +
           `<div class="meta">${d.pages}p · ${d.chunks} chunks</div></div></div>`;
  }).join("");
  scope.innerHTML = '<option value="">All documents</option>' +
    ids.map((id) => `<option value="${id}">${esc(state.docs[id].name)}</option>`).join("");
  scope.value = state.scope;
}

// ---- chat -----------------------------------------------------------------
function msgEl(role, inner) {
  hideWelcome();
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  wrap.innerHTML = `<div class="ava">${role === "user" ? "U" : "◈"}</div><div class="bubble">${inner}</div>`;
  $("messages").appendChild(wrap);
  scrollDown();
  return wrap;
}

function botMsg(inner) {
  return `<div class="msg bot"><div class="ava">◈</div><div class="bubble">${inner}</div></div>`;
}

function resultInner(r) {
  const status = r.status || "answered";
  const label = { answered: "answered", summary: "summary", insufficient_evidence: "no answer in docs",
                  smalltalk: "assistant", images: "image extraction" }[status] || status;
  const pillClass = status === "answered" && r.intent === "summarize" ? "summary" : status;
  let html = `<span class="pill ${pillClass}">${label}</span>${esc(r.answer)}`;
  if (r.sources && r.sources.length) {
    html += `<div class="sources">` + r.sources.map((s) =>
      `<span class="src"><b>${esc(docName(s.doc_id))}</b> · p.${s.page_number}</span>`).join("") + `</div>`;
  }
  if (r.trace && r.trace.length) {
    html += `<details class="trace"><summary>agent steps</summary><pre>${esc(JSON.stringify(r.trace, null, 2))}</pre></details>`;
  }
  return html;
}

async function send(preset) {
  const input = $("input");
  const text = (preset || input.value).trim();
  if (!text || state.busy) return;
  if (!preset) { input.value = ""; input.style.height = "auto"; }
  state.busy = true; $("send").disabled = true;

  msgEl("user", esc(text));
  state.messages.push({ role: "user", content: text });
  const typing = msgEl("bot", `<span class="typing"><i></i><i></i><i></i></span>`);

  try {
    const r = await api("/agent", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: state.messages, mode: state.mode, doc_id: state.scope || null }),
    });
    typing.outerHTML = botMsg(resultInner(r));
    state.messages.push({ role: "assistant", content: r.answer || "" });
  } catch (e) {
    const hint = /unavailable|503/i.test(e.message)
      ? "The local model isn't reachable. Start it (llama.cpp or Ollama) and try again."
      : e.message;
    typing.outerHTML = botMsg(`<span class="pill error">error</span>${esc(hint)}`);
  } finally {
    state.busy = false; $("send").disabled = false; scrollDown();
  }
}

// ---- wiring ---------------------------------------------------------------
function wireDrop() {
  const drop = $("drop"), file = $("file");
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
document.querySelectorAll(".chip").forEach((c) =>
  c.addEventListener("click", () => { $("input").value = c.dataset.q; $("input").focus();
    $("input").style.height = "auto"; $("input").style.height = $("input").scrollHeight + "px"; }));
$("scope").addEventListener("change", (e) => { state.scope = e.target.value; });
$("send").addEventListener("click", () => send());
$("input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
$("input").addEventListener("input", (e) => { e.target.style.height = "auto"; e.target.style.height = e.target.scrollHeight + "px"; });
wireDrop();
