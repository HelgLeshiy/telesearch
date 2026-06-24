"use strict";
const API = "/api";
let token = localStorage.getItem("ts_token") || "";
let mode = "login";
let workspaceId = "";
let jobsTimer = null;

const $ = (id) => document.getElementById(id);

async function api(path, { method = "GET", body, form } = {}) {
  const headers = {};
  if (token) headers["Authorization"] = "Bearer " + token;
  let payload;
  if (form) {
    payload = form;
  } else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const res = await fetch(API + path, { method, headers, body: payload });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  if (res.status === 204) return null;
  return res.json();
}

// ---- Auth ----
document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    mode = t.dataset.mode;
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    $("auth-name").classList.toggle("hidden", mode !== "register");
    $("auth-submit").textContent = mode === "register" ? "Register" : "Log in";
  })
);

$("auth-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("auth-error").textContent = "";
  const payload = { email: $("auth-email").value, password: $("auth-password").value };
  if (mode === "register") payload.name = $("auth-name").value;
  try {
    const data = await api("/auth/" + mode, { method: "POST", body: payload });
    token = data.access_token;
    localStorage.setItem("ts_token", token);
    await boot();
  } catch (err) {
    $("auth-error").textContent = err.message;
  }
});

$("logout").addEventListener("click", () => {
  token = "";
  localStorage.removeItem("ts_token");
  if (jobsTimer) clearInterval(jobsTimer);
  showAuth();
});

function showAuth() {
  $("auth").classList.remove("hidden");
  $("app").classList.add("hidden");
  $("user-bar").classList.add("hidden");
}

async function boot() {
  try {
    const me = await api("/auth/me");
    $("user-email").textContent = me.email;
  } catch (e) {
    showAuth();
    return;
  }
  $("auth").classList.add("hidden");
  $("app").classList.remove("hidden");
  $("user-bar").classList.remove("hidden");
  await loadWorkspaces();
  await refresh();
  await loadGuides();
  await loadPresets();
  await loadGraph();
  if (jobsTimer) clearInterval(jobsTimer);
  jobsTimer = setInterval(refresh, 3000);
}

// ---- Knowledge graph ----
let graphData = { nodes: [], edges: [] };

async function loadGraph() {
  if (!workspaceId) return;
  try {
    graphData = await api(`/workspaces/${workspaceId}/graph`);
    drawGraph();
    const n = (graphData.meta && graphData.meta.n_topics) || 0;
    $("graph-status").textContent = n
      ? `${n} topics across ${graphData.meta.n_chunks} chunks. Click a node to explore.`
      : "No graph yet — index some sources, then Rebuild graph.";
  } catch (e) {}
}

function _palette(i) {
  const colors = ["#5b8cff", "#ff8c5b", "#5bffa0", "#d65bff", "#ffd65b", "#5bd6ff", "#ff5b8c"];
  return colors[i % colors.length];
}

function drawGraph() {
  const cv = $("graph-canvas");
  const ctx = cv.getContext("2d");
  const W = (cv.width = cv.clientWidth), H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const nodes = graphData.nodes || [];
  if (!nodes.length) return;
  const pad = 50;
  const sx = (x) => pad + ((x + 1) / 2) * (W - 2 * pad);
  const sy = (y) => pad + ((y + 1) / 2) * (H - 2 * pad);
  const maxSize = Math.max(...nodes.map((n) => n.size), 1);
  const radius = (n) => 8 + 26 * Math.sqrt(n.size / maxSize);
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));

  (graphData.edges || []).forEach((e) => {
    const a = byId[e.source], b = byId[e.target];
    if (!a || !b) return;
    ctx.beginPath();
    ctx.moveTo(sx(a.x), sy(a.y));
    ctx.lineTo(sx(b.x), sy(b.y));
    ctx.strokeStyle = `rgba(150,160,180,${0.15 + 0.5 * (e.weight || 0)})`;
    ctx.lineWidth = 1 + 2 * (e.weight || 0);
    ctx.stroke();
  });

  nodes.forEach((n, i) => {
    const r = radius(n), x = sx(n.x), y = sy(n.y);
    n._sx = x; n._sy = y; n._r = r;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = _palette(i);
    ctx.globalAlpha = 0.85;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#e7e9ee";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText((n.label || "").slice(0, 24), x, y + r + 13);
  });
}

$("graph-canvas").addEventListener("click", (e) => {
  const cv = $("graph-canvas");
  const rect = cv.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const hit = (graphData.nodes || []).find(
    (n) => Math.hypot(n._sx - mx, n._sy - my) <= (n._r || 0)
  );
  if (hit) showTopic(hit);
});

function showTopic(node) {
  const p = $("topic-panel");
  p.classList.remove("hidden");
  p.innerHTML =
    `<h4>${esc(node.label)}</h4>` +
    `<div class="muted">${node.size} chunks · keywords: ${esc((node.keywords || []).join(", "))}</div>` +
    (node.sample_contents || [])
      .map((s) => `<div class="sample">${esc(s)}</div>`)
      .join("");
}

$("graph-refresh").addEventListener("click", async () => {
  if (!workspaceId) return;
  $("graph-status").textContent = "Rebuilding graph...";
  try {
    await api(`/workspaces/${workspaceId}/graph/refresh`, { method: "POST" });
    let tries = 0;
    const poll = setInterval(async () => {
      tries++;
      await loadGraph();
      if ((graphData.nodes || []).length || tries > 20) clearInterval(poll);
    }, 1500);
  } catch (err) {
    $("graph-status").textContent = "Error: " + err.message;
  }
});

async function loadGuides() {
  try {
    const guides = await api("/guides");
    $("guides").innerHTML = guides
      .map(
        (g) =>
          `<div class="guide"><h4>${esc(g.title)}</h4><ol>` +
          g.steps.map((s) => `<li>${esc(s)}</li>`).join("") +
          `</ol></div>`
      )
      .join("");
  } catch (e) {}
}

async function loadWorkspaces() {
  const wss = await api("/workspaces");
  const sel = $("workspace-select");
  sel.innerHTML = "";
  wss.forEach((w) => {
    const o = document.createElement("option");
    o.value = w.id;
    o.textContent = `${w.name} (${w.role})`;
    sel.appendChild(o);
  });
  workspaceId = wss.length ? wss[0].id : "";
  sel.onchange = () => { workspaceId = sel.value; refresh(); loadGraph(); };
}

$("delete-ws").addEventListener("click", async () => {
  if (!workspaceId) return;
  if (!confirm("Permanently delete this workspace and ALL its data?")) return;
  try {
    await api(`/workspaces/${workspaceId}`, { method: "DELETE" });
    await loadWorkspaces();
    await refresh();
    await loadGraph();
  } catch (err) {
    alert("Delete failed: " + err.message);
  }
});

// ---- Sources + jobs ----
$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!workspaceId) return;
  const fd = new FormData();
  fd.append("file", $("file").files[0]);
  fd.append("name", $("src-name").value);
  fd.append("kind", $("src-kind").value);
  fd.append("index_media", $("src-media").checked ? "true" : "false");
  $("upload-status").textContent = "Uploading...";
  try {
    await api(`/workspaces/${workspaceId}/sources`, { method: "POST", form: fd });
    $("upload-status").textContent = "Uploaded. Indexing in background.";
    $("upload-form").reset();
    refresh();
  } catch (err) {
    $("upload-status").textContent = "Error: " + err.message;
  }
});

async function refresh() {
  if (!workspaceId) return;
  try {
    const [sources, jobs] = await Promise.all([
      api(`/workspaces/${workspaceId}/sources`),
      api(`/workspaces/${workspaceId}/jobs`),
    ]);
    renderSources(sources);
    renderJobs(jobs);
  } catch (e) {}
}

function renderSources(sources) {
  const tb = $("sources-table").querySelector("tbody");
  tb.innerHTML = "";
  sources.forEach((s) => {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${esc(s.name)}</td><td>${esc(s.kind || "auto")}</td>` +
      `<td><span class="badge">${s.status}</span></td>` +
      `<td>${(s.bytes / 1024).toFixed(1)} KB</td>` +
      `<td><button class="danger" data-id="${s.id}">Delete</button></td>`;
    tr.querySelector("button").onclick = async () => {
      await api(`/workspaces/${workspaceId}/sources/${s.id}`, { method: "DELETE" });
      refresh();
    };
    tb.appendChild(tr);
  });
}

function renderJobs(jobs) {
  const ul = $("jobs-list");
  ul.innerHTML = "";
  jobs.slice(0, 8).forEach((j) => {
    const li = document.createElement("li");
    const pct = Math.round((j.progress || 0) * 100);
    li.textContent = `${j.type} — ${j.state} (${pct}%) ${j.message || j.error || ""}`;
    ul.appendChild(li);
  });
}

// ---- Search ----
function currentSearchBody() {
  const body = { query: $("q").value, k: 10 };
  const mod = $("f-modality").value.trim();
  if (mod) body.modalities = [mod];
  const kind = $("f-kind").value.trim();
  if (kind) body.source_kinds = [kind];
  const since = $("f-since").value, until = $("f-until").value;
  if (since) body.date_from = Math.floor(new Date(since).getTime() / 1000);
  if (until) body.date_to = Math.floor(new Date(until).getTime() / 1000);
  return body;
}

$("search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = currentSearchBody();
  const global = $("global-toggle").checked;
  if (!global && !workspaceId) return;
  $("results").innerHTML = "<p class='muted'>Searching...</p>";
  try {
    const path = global ? "/search" : `/workspaces/${workspaceId}/search`;
    const hits = await api(path, { method: "POST", body });
    renderResults(hits);
  } catch (err) {
    $("results").innerHTML = `<p class='error'>${esc(err.message)}</p>`;
  }
});

// ---- Presets ----
async function loadPresets() {
  try {
    const presets = await api("/presets");
    const sel = $("preset-select");
    sel.innerHTML = '<option value="">— saved searches —</option>';
    presets.forEach((p) => {
      const o = document.createElement("option");
      o.value = p.id;
      o.textContent = p.name;
      o.dataset.params = JSON.stringify(p.params || {});
      sel.appendChild(o);
    });
  } catch (e) {}
}

$("preset-select").addEventListener("change", (e) => {
  const opt = e.target.selectedOptions[0];
  if (!opt || !opt.dataset.params) return;
  const p = JSON.parse(opt.dataset.params);
  $("q").value = p.query || "";
  $("f-modality").value = (p.modalities || [])[0] || "";
  $("f-kind").value = (p.source_kinds || [])[0] || "";
  $("global-toggle").checked = !!p.global;
});

$("preset-save").addEventListener("click", async () => {
  const name = $("preset-name").value.trim();
  if (!name) return;
  const params = currentSearchBody();
  params.global = $("global-toggle").checked;
  await api("/presets", { method: "POST", body: { name, params } });
  $("preset-name").value = "";
  loadPresets();
});

$("preset-delete").addEventListener("click", async () => {
  const id = $("preset-select").value;
  if (!id) return;
  await api(`/presets/${id}`, { method: "DELETE" });
  loadPresets();
});

function renderResults(hits) {
  const box = $("results");
  if (!hits.length) { box.innerHTML = "<p class='muted'>No results.</p>"; return; }
  box.innerHTML = "";
  hits.forEach((h) => {
    const div = document.createElement("div");
    div.className = "result";
    div.innerHTML =
      `<div class="meta">${esc(h.modality)} · ${esc(h.sender)} · ${esc(h.date_str)} · score ${h.score.toFixed(3)}</div>` +
      `<div class="content">${esc(h.content)}</div>`;
    box.appendChild(div);
  });
}

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
  );
}

boot();
