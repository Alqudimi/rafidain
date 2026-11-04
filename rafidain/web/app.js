/* رافدَين — client app: realtime mic streaming + file conversion + management.
 * Vanilla JS, no build step. Talks to the same FastAPI backend that serves it.
 */
"use strict";

/* ------------------------------------------------------------------ state */
const S = {
  apiKey: localStorage.getItem("rafidain_api_key") || "",
  connected: false,
  voices: [],
  sessionInfo: null,
  ws: null,
  streaming: false,
  paused: false,
  captureCtx: null,
  playbackCtx: null,
  captureNode: null,
  playbackNode: null,
  micStream: null,
  inRate: 16000,
  outRate: 16000,
  blockMs: 180,
  stats: { t0: 0, blocksIn: 0, blocksOut: 0, sentBytes: 0, recvBytes: 0, drops: 0 },
  metricsTimer: null,
  lastMetrics: null,
};

const $ = (id) => document.getElementById(id);
const el = {
  key: $("api-key"), keyStatus: $("key-status"),
  chipServer: $("chip-server"), chipEngine: $("chip-engine"), chipStream: $("chip-stream"),
  liveVoice: $("live-voice"), liveProfile: $("live-profile"), micSelect: $("mic-select"),
  liveMonitor: $("live-monitor"),
  btnStart: $("btn-start"), btnPause: $("btn-pause"), btnStop: $("btn-stop"),
  meterIn: $("meter-in"), meterOut: $("meter-out"),
  stUptime: $("st-uptime"), stBlocks: $("st-blocks"), stRtf: $("st-rtf"),
  stLatency: $("st-latency"), stQueued: $("st-out-queued"), stLost: $("st-lost"),
  liveLog: $("live-log"), fileLog: $("file-log"),
  fileSource: $("file-source"), sourceName: $("source-name"), sourceAudio: $("source-audio"),
  dropSource: $("drop-source"),
  fileVoice: $("file-voice"), fileReference: $("file-reference"), referenceName: $("reference-name"),
  fileProfile: $("file-profile"), btnConvert: $("btn-convert"),
  convertProgress: $("convert-progress"),
  resultPlaceholder: $("result-placeholder"), resultBox: $("result-box"),
  resultAudio: $("result-audio"), resultDownload: $("result-download"),
  cvSeconds: $("cv-seconds"), cvElapsed: $("cv-elapsed"), cvRtf: $("cv-rtf"), cvRate: $("cv-rate"),
  voiceName: $("voice-name"), voiceFile: $("voice-file"), voiceFileName: $("voice-file-name"),
  voiceAudio: $("voice-audio"), btnRegisterVoice: $("btn-register-voice"),
  voiceList: $("voice-list"), sessionList: $("session-list"), sysinfo: $("sysinfo"),
  jobId: $("job-id"), jobInfo: $("job-info"), toast: $("toast"),
};

/* ------------------------------------------------------------- utilities */
function toast(msg, kind = "", ms = 3200) {
  el.toast.textContent = msg;
  el.toast.className = "toast " + kind;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (el.toast.className = "toast hidden"), ms);
}

function log(box, msg, cls = "") {
  const line = document.createElement("div");
  if (cls) line.className = cls;
  line.textContent = `[${new Date().toLocaleTimeString("en-GB")}] ${msg}`;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

const fmtBytes = (b) => (b > 1048576 ? (b / 1048576).toFixed(1) + "MB" : (b / 1024).toFixed(1) + "KB");
const fmtDur = (s) => {
  s = Math.max(0, Math.round(s));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
};

function setChip(chip, text, cls) {
  chip.querySelector(".dot");
  chip.childNodes[chip.childNodes.length - 1].textContent = " " + text;
  chip.className = "chip " + (cls || "");
}

/* linear resample (voice-grade is fine for UI playback paths) */
function resampleLinear(f32, fromRate, toRate) {
  if (fromRate === toRate || !f32.length) return f32;
  const ratio = toRate / fromRate;
  const outLen = Math.max(1, Math.round(f32.length * ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const x = i / ratio;
    const i0 = Math.floor(x), i1 = Math.min(i0 + 1, f32.length - 1);
    const fr = x - i0;
    out[i] = f32[i0] * (1 - fr) + f32[i1] * fr;
  }
  return out;
}

const f32ToI16 = (f32) => {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1 - 1e-6, f32[i]));
    out[i] = s < 0 ? s * 32768 : s * 32767;
  }
  return out;
};
const i16ToF32 = (i16) => {
  const out = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) out[i] = i16[i] / 32768;
  return out;
};

/* ------------------------------------------------------------------- API */
async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (S.apiKey) headers["X-API-Key"] = S.apiKey;
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    opts = Object.assign({}, opts, { body: JSON.stringify(opts.json) });
  }
  const resp = await fetch(path, Object.assign({}, opts, { headers }));
  if (resp.status === 401) {
    throw Object.assign(new Error("مفتاح API مطلوب أو غير صالح (401)"), { status: 401 });
  }
  const text = await resp.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text }; }
  if (!resp.ok) {
    const msg = (data && data.error && data.error.message) || (data && data.detail) || `HTTP ${resp.status}`;
    throw Object.assign(new Error(msg), { status: resp.status, data });
  }
  return data;
}

/* --------------------------------------------------------------- connect */
async function connect() {
  S.apiKey = el.key.value.trim();
  localStorage.setItem("rafidain_api_key", S.apiKey);
  el.keyStatus.textContent = "جارٍ الاتصال…";
  try {
    const health = await api("/healthz");
    setChip(el.chipServer, `متصل · v${health.version || "?"}`, "ok");
    let ready = null;
    try { ready = await api("/readyz"); } catch { ready = null; }
    if (ready) {
      const eng = ready.engine || {};
      setChip(
        el.chipEngine,
        eng.impl === "mock" ? "محرك وهمي (تجريبي)" : `محرك ${eng.impl}: ${eng.ready ? "جاهز" : "قيد التحميل"}`,
        eng.ready ? "ok" : "warn"
      );
      el.sysinfo.textContent = JSON.stringify(ready, null, 2);
    } else {
      setChip(el.chipEngine, "المحرك: غير معروف", "warn");
    }
    S.connected = true;
    el.keyStatus.textContent = "✓ متصل";
    el.keyStatus.className = "muted";
    await refreshVoices();
    log(el.liveLog, `connected — engine ready (${ready ? (ready.engine || {}).impl : "?"})`, "ok");
  } catch (err) {
    S.connected = false;
    setChip(el.chipServer, "غير متصل", "err");
    setChip(el.chipEngine, "—", "");
    el.keyStatus.textContent = "✗ فشل الاتصال";
    el.keyStatus.className = "muted";
    toast(err.message, "err");
    log(el.liveLog, "connection failed: " + err.message, "err");
  }
}

async function refreshMics() {
  try {
    let devs = [];
    try { devs = await navigator.mediaDevices.enumerateDevices(); }
    catch { await navigator.mediaDevices.getUserMedia({ audio: true }); devs = await navigator.mediaDevices.enumerateDevices(); }
    const mics = devs.filter((d) => d.kind === "audioinput");
    const cur = el.micSelect.value;
    el.micSelect.innerHTML = "";
    mics.forEach((m, i) => {
      const o = document.createElement("option");
      o.value = m.deviceId;
      o.textContent = m.label || `ميكروفون ${i + 1}`;
      el.micSelect.appendChild(o);
    });
    if (cur) el.micSelect.value = cur;
    if (!mics.length) el.micSelect.innerHTML = "<option>لا يوجد ميكروفون</option>";
  } catch (err) { toast("تعذّر الوصول للميكروفون: " + err.message, "err"); }
}

async function refreshVoices() {
  try {
    const data = await api("/api/v1/voices");
    S.voices = data.voices || [];
    for (const sel of [el.liveVoice, el.fileVoice]) {
      const cur = sel.value;
      sel.innerHTML = "";
      if (!S.voices.length) {
        sel.innerHTML = "<option value=''>— سجّل صوتاً مستهدفاً أولاً (تبويب الأصوات) —</option>";
      } else {
        S.voices.forEach((v) => {
          const o = document.createElement("option");
          o.value = v.voice_id;
          o.textContent = `${v.name || v.voice_id} (${(v.duration_s || 0).toFixed(1)}ث)`;
          sel.appendChild(o);
        });
        if (cur) sel.value = cur;
      }
    }
    renderVoiceList();
    updateConvertBtn();
  } catch (err) {
    if (err.status !== 401) toast("تعذّر جلب الأصوات: " + err.message, "err");
  }
}

function renderVoiceList() {
  el.voiceList.innerHTML = "";
  if (!S.voices.length) {
    el.voiceList.innerHTML = "<li class='muted'>لا توجد أصوات مسجَّلة</li>";
    return;
  }
  S.voices.forEach((v) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <span><span class="vname">${escapeHtml(v.name || v.voice_id)}</span>
        <span class="badge">${(v.duration_s || 0).toFixed(1)}ث</span></span>
      <span class="vmeta">${v.voice_id}</span>
      <span class="row">
        <button class="btn small-btn" data-use="${v.voice_id}">استخدام</button>
        <button class="btn small-btn danger" data-del="${v.voice_id}">حذف</button>
      </span>`;
    el.voiceList.appendChild(li);
  });
  el.voiceList.querySelectorAll("[data-del]").forEach((b) =>
    b.addEventListener("click", async () => {
      if (!confirm("حذف هذا الصوت؟")) return;
      try { await api(`/api/v1/voices/${b.dataset.del}`, { method: "DELETE" }); toast("حُذف الصوت", "ok"); await refreshVoices(); }
      catch (err) { toast(err.message, "err"); }
    })
  );
  el.voiceList.querySelectorAll("[data-use]").forEach((b) =>
    b.addEventListener("click", () => {
      el.liveVoice.value = b.dataset.use;
      el.fileVoice.value = b.dataset.use;
      document.querySelector('.tab[data-tab="live"]').click();
      toast("تم اختيار الصوت المستهدف ✓", "ok", 1600);
    })
  );
}
const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ==================================================== LIVE STREAMING ==== */
async function startLive() {
  if (!S.connected) { toast("اتصل بالخادم أولاً", "err"); return; }
  const voiceId = el.liveVoice.value;
  if (!voiceId) { toast("اختر صوتاً مستهدفاً أولاً (سجّل واحداً من تبويب الأصوات)", "err"); return; }
  disableControls(true);
  try {
    // 1) create session
    S.sessionInfo = await api("/api/v1/sessions", {
      method: "POST", json: { voice_id: voiceId, latency_profile: el.liveProfile.value },
    });
    log(el.liveLog, `session ${S.sessionInfo.session_id} (profile=${S.sessionInfo.profile.key}, block=${S.sessionInfo.profile.block_ms}ms)`);

    // 2) websocket
    const wsUrl = new URL(S.sessionInfo.ws_url);
    if (location.protocol === "https:") wsUrl.protocol = "wss:";
    else wsUrl.protocol = wsUrl.protocol === "https:" ? "wss:" : "ws:";
    if (S.apiKey) wsUrl.searchParams.set("token", S.apiKey);
    const ws = new WebSocket(wsUrl.toString());
    ws.binaryType = "arraybuffer";
    S.ws = ws;

    ws.onopen = () => log(el.liveLog, "websocket open");
    ws.onerror = () => log(el.liveLog, "websocket error", "err");
    ws.onclose = (ev) => {
      log(el.liveLog, `websocket closed (${ev.code}${ev.reason ? " · " + ev.reason : ""})`, S.streaming ? "err" : "");
      if (S.streaming || S.paused) teardownAudio("انقطع الاتصال");
    };
    ws.onmessage = onWsMessage;
  } catch (err) {
    toast(err.message, "err");
    log(el.liveLog, "start failed: " + err.message, "err");
    disableControls(false);
  }
}

async function onWsMessage(ev) {
  if (typeof ev.data === "string") {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    switch (msg.type) {
      case "session.ready":
        S.inRate = (msg.audio_in && msg.audio_in.sample_rate) || 16000;
        S.outRate = (msg.audio_out && msg.audio_out.sample_rate) || 16000;
        S.blockMs = (msg.profile && msg.profile.block_ms) || 180;
        log(el.liveLog, `ready: in=${S.inRate}Hz out=${S.outRate}Hz block=${S.blockMs}ms`, "ok");
        await setupAudio();
        S.ws.send(JSON.stringify({ type: "stream.start" }));
        break;
      case "stream.started":
        S.streaming = true; S.paused = false;
        S.stats = { t0: performance.now(), blocksIn: 0, blocksOut: 0, sentBytes: 0, recvBytes: 0, drops: 0 };
        setChip(el.chipStream, "البث جارٍ 🔴", "ok");
        disableControls(false);
        log(el.liveLog, "stream started — تحدث الآن 🎙", "ok");
        S.metricsTimer = setInterval(() => {
          if (S.ws && S.ws.readyState === 1) S.ws.send(JSON.stringify({ type: "metrics.poll" }));
        }, 2000);
        break;
      case "stream.paused":
        S.paused = true;
        setChip(el.chipStream, "متوقف مؤقتاً", "warn");
        break;
      case "stream.resumed":
        S.paused = false;
        setChip(el.chipStream, "البث جارٍ 🔴", "ok");
        break;
      case "stream.stopped":
        log(el.liveLog, "stream stopped");
        break;
      case "metrics.snapshot":
        S.lastMetrics = msg.metrics || {};
        updateMetricsUI();
        break;
      case "warning":
        log(el.liveLog, "warning: " + (msg.message || msg.code), "warn");
        break;
      case "error":
        log(el.liveLog, "error: " + (msg.message || msg.code), "err");
        toast("خطأ من الخادم: " + (msg.message || msg.code), "err");
        break;
      case "session.closed":
        log(el.liveLog, "session closed: " + (msg.reason || ""));
        break;
    }
    return;
  }
  // binary: converted PCM s16le @ S.outRate
  const i16 = new Int16Array(ev.data);
  S.stats.recvBytes += ev.data.byteLength;
  S.stats.blocksOut++;
  if (el.liveMonitor.checked && S.playbackNode) {
    let f32 = i16ToF32(i16);
    if (S.playbackCtx.sampleRate !== S.outRate) f32 = resampleLinear(f32, S.outRate, S.playbackCtx.sampleRate);
    S.playbackNode.port.postMessage({ pcm: f32 });
    meter(el.meterOut, f32);
  } else {
    meter(el.meterOut, i16ToF32(i16));
  }
}

async function setupAudio() {
  // ---- microphone capture
  const micId = el.micSelect.value;
  const constraints = {
    audio: Object.assign(
      { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
      micId ? { deviceId: { exact: micId } } : {}
    ),
  };
  S.micStream = await navigator.mediaDevices.getUserMedia(constraints);
  try { S.captureCtx = new AudioContext({ sampleRate: S.inRate }); }
  catch { S.captureCtx = new AudioContext(); }
  const capCtxRate = S.captureCtx.sampleRate;
  await S.captureCtx.resume();
  await S.captureCtx.audioWorklet.addModule("pcm-worklet.js");
  const blockSize = Math.max(128, Math.round((S.blockMs / 1000) * capCtxRate));
  S.captureNode = new AudioWorkletNode(S.captureCtx, "rafidain-capture", {
    processorOptions: { blockSize }, numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1,
  });
  S.captureNode.port.onmessage = (e) => {
    if (!S.ws || S.ws.readyState !== 1 || !S.streaming || S.paused) return;
    let f32 = new Float32Array(e.data);
    meter(el.meterIn, f32);
    if (capCtxRate !== S.inRate) f32 = resampleLinear(f32, capCtxRate, S.inRate);
    const i16 = f32ToI16(f32);
    S.ws.send(i16.buffer);
    S.stats.sentBytes += i16.byteLength;
    S.stats.blocksIn++;
  };
  S.captureCtx.createMediaStreamSource(S.micStream).connect(S.captureNode);
  S.captureNode.connect(S.captureCtx.createGain()); // worklet needs a sink

  // ---- converted playback
  if (el.liveMonitor.checked) {
    try { S.playbackCtx = new AudioContext({ sampleRate: S.outRate }); }
    catch { S.playbackCtx = new AudioContext(); }
    await S.playbackCtx.resume();
    await S.playbackCtx.audioWorklet.addModule("pcm-worklet.js");
    S.playbackNode = new AudioWorkletNode(S.playbackCtx, "rafidain-playback", {
      numberOfInputs: 0, numberOfOutputs: 1, channelCount: 1, outputChannelCount: [1],
    });
    S.playbackNode.connect(S.playbackCtx.destination);
  }
  refreshMics().catch(() => {});
}

function teardownAudio(reason) {
  clearInterval(S.metricsTimer);
  S.metricsTimer = null;
  S.streaming = false; S.paused = false;
  try { if (S.ws) { S.ws.onclose = null; S.ws.close(); } } catch { /* noop */ }
  S.ws = null;
  if (S.micStream) { S.micStream.getTracks().forEach((t) => t.stop()); S.micStream = null; }
  for (const ctx of [S.captureCtx, S.playbackCtx]) { try { ctx && ctx.close(); } catch { /* noop */ } }
  S.captureCtx = S.playbackCtx = S.captureNode = S.playbackNode = null;
  setChip(el.chipStream, "لا يوجد بث", "");
  disableControls(false);
  el.meterIn.style.width = "0%"; el.meterOut.style.width = "0%";
  if (reason) log(el.liveLog, reason, "warn");
}

function pauseLive() {
  if (!S.streaming) return;
  if (!S.paused) {
    S.ws && S.ws.send(JSON.stringify({ type: "stream.pause" }));
    S.captureNode && S.captureNode.port.postMessage({ cmd: "mute", value: true });
    el.btnPause.textContent = "▶ استئناف";
  } else {
    S.ws && S.ws.send(JSON.stringify({ type: "stream.resume" }));
    S.captureNode && S.captureNode.port.postMessage({ cmd: "mute", value: false });
    el.btnPause.textContent = "⏸ إيقاف مؤقت";
  }
}

function stopLive() {
  if (!S.streaming && !S.paused) { teardownAudio(); return; }
  try { S.ws && S.ws.send(JSON.stringify({ type: "stream.stop" })); } catch { /* noop */ }
  setTimeout(() => teardownAudio(), 300);
}

function meter(barEl, f32) {
  let sum = 0;
  for (let i = 0; i < f32.length; i += 16) sum += f32[i] * f32[i];
  const rms = Math.sqrt(sum / Math.max(1, Math.ceil(f32.length / 16)));
  const pct = Math.min(100, rms * 180);
  barEl.style.width = pct.toFixed(0) + "%";
}

function updateMetricsUI() {
  const m = S.lastMetrics || {};
  const up = (performance.now() - S.stats.t0) / 1000;
  el.stUptime.textContent = fmtDur(up);
  el.stBlocks.textContent = `${S.stats.blocksIn}/${S.stats.blocksOut}`;
  el.stRtf.textContent = m.rtf != null ? Number(m.rtf).toFixed(2) : (m.block_rtf != null ? Number(m.block_rtf).toFixed(2) : "—");
  el.stLatency.textContent = m.block_seconds != null ? (m.block_seconds * 1000).toFixed(0) + "ms" : (m.block_ms != null ? m.block_ms + "ms" : "—");
  el.stQueued.textContent = S.playbackNode ? (m.out_queue || "—") : "معطّل";
  el.stLost.textContent = String(m.overflows || m.drops || S.stats.drops || 0);
}

function disableControls(started) {
  el.btnStart.disabled = started || S.streaming || S.paused;
  el.btnPause.disabled = !(started ? true : S.streaming || S.paused);
  el.btnStop.disabled = !(started ? true : S.streaming || S.paused);
  if (!started && !S.streaming && !S.paused) el.btnPause.textContent = "⏸ إيقاف مؤقت";
}

/* ==================================================== FILE CONVERSION === */
let sourceFile = null, referenceFile = null;

function updateConvertBtn() {
  const targetMode = document.querySelector('input[name="file-target"]:checked').value;
  const voiceOk = targetMode === "voice" && !!el.fileVoice.value;
  const refOk = targetMode === "reference" && !!referenceFile;
  el.btnConvert.disabled = !(sourceFile && (voiceOk || refOk));
}

async function convertFile() {
  if (!sourceFile) return;
  const targetMode = document.querySelector('input[name="file-target"]:checked').value;
  const fd = new FormData();
  fd.append("source", sourceFile, sourceFile.name);
  fd.append("latency_profile", el.fileProfile.value);
  if (targetMode === "voice") fd.append("voice_id", el.fileVoice.value);
  else fd.append("reference", referenceFile, referenceFile.name);

  el.btnConvert.disabled = true;
  el.convertProgress.classList.remove("hidden");
  const bar = el.convertProgress.querySelector("i");
  const lbl = el.convertProgress.querySelector("span");
  bar.style.width = "0%"; lbl.textContent = "رفع الملف… ٠٪";
  log(el.fileLog, `converting ${sourceFile.name} (${fmtBytes(sourceFile.size)}) → ${targetMode === "voice" ? el.fileVoice.value : "inline reference"} [${el.fileProfile.value}]`);

  const t0 = performance.now();
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/v1/convert");
  if (S.apiKey) xhr.setRequestHeader("X-API-Key", S.apiKey);
  xhr.responseType = "blob";
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 100);
      bar.style.width = pct + "%";
      lbl.textContent = pct < 100 ? `رفع الملف… ${pct}٪` : "جارٍ التحويل…";
    }
  };
  xhr.onload = () => {
    el.btnConvert.disabled = false;
    if (xhr.status !== 200) {
      el.convertProgress.classList.add("hidden");
      const err = new Error(`HTTP ${xhr.status}`);
      xhr.response.text().then((t) => {
        try { err.message = JSON.parse(t).error.message || t; } catch { err.message = t.slice(0, 200); }
        toast("فشل التحويل: " + err.message, "err", 6000);
        log(el.fileLog, "convert failed: " + err.message, "err");
      });
      return;
    }
    bar.style.width = "100%";
    lbl.textContent = "تم ✓";
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    const h = xhr.getAllResponseHeaders();
    el.cvSeconds.textContent = (xhr.getResponseHeader("X-Conversion-Seconds") || "—") + "ث";
    el.cvElapsed.textContent = elapsed + "ث";
    el.cvRtf.textContent = xhr.getResponseHeader("X-Conversion-RTF") || "—";
    el.cvRate.textContent = (xhr.getResponseHeader("X-Conversion-Sample-Rate") || "—") + "Hz";
    const url = URL.createObjectURL(xhr.response);
    el.resultAudio.src = url;
    el.resultDownload.href = url;
    el.resultPlaceholder.classList.add("hidden");
    el.resultBox.classList.remove("hidden");
    log(el.fileLog, `converted in ${elapsed}s (server rtf=${xhr.getResponseHeader("X-Conversion-RTF")})`, "ok");
    toast("تم التحويل ✓ استمع أو نزّل النتيجة", "ok");
    setTimeout(() => el.convertProgress.classList.add("hidden"), 1500);
    void h;
  };
  xhr.onerror = () => {
    el.btnConvert.disabled = false;
    el.convertProgress.classList.add("hidden");
    toast("خطأ شبكة أثناء التحويل", "err");
    log(el.fileLog, "network error during convert", "err");
  };
  xhr.send(fd);
}

/* ========================================================= OPS PANEL === */
async function refreshSessions() {
  try {
    const data = await api("/api/v1/sessions");
    el.sessionList.innerHTML = "";
    const list = data.sessions || [];
    if (!list.length) { el.sessionList.innerHTML = "<li class='muted'>لا توجد جلسات</li>"; return; }
    list.forEach((s) => {
      const li = document.createElement("li");
      const live = ["streaming", "ready", "created", "paused"].includes(s.state);
      li.innerHTML = `
        <span><span class="badge ${live ? "live" : ""}">${s.state}</span> ${s.voice_id || ""}</span>
        <span class="vmeta">${s.session_id}</span>
        <span class="row"><button class="btn small-btn danger" data-close="${s.session_id}">إغلاق</button></span>`;
      el.sessionList.appendChild(li);
    });
    el.sessionList.querySelectorAll("[data-close]").forEach((b) =>
      b.addEventListener("click", async () => {
        try { await api(`/api/v1/sessions/${b.dataset.close}`, { method: "DELETE" }); refreshSessions(); }
        catch (err) { toast(err.message, "err"); }
      })
    );
  } catch (err) { if (err.status !== 401) log(el.liveLog, "sessions: " + err.message, "warn"); }
}

async function jobGet() {
  const id = el.jobId.value.trim();
  if (!id) { toast("أدخل معرّف المهمة", "err"); return; }
  try {
    const data = await api(`/api/v1/finetune/jobs/${encodeURIComponent(id)}`);
    el.jobInfo.textContent = JSON.stringify(data, null, 2);
  } catch (err) { el.jobInfo.textContent = "خطأ: " + err.message; }
}

async function jobCancel() {
  const id = el.jobId.value.trim();
  if (!id) { toast("أدخل معرّف المهمة", "err"); return; }
  try {
    const data = await api(`/api/v1/finetune/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
    el.jobInfo.textContent = JSON.stringify(data, null, 2);
    toast("أُلغيت المهمة (إن كانت قابلة للإلغاء)", "ok");
  } catch (err) { toast(err.message, "err"); }
}

async function showInfo() {
  try {
    const ready = await api("/readyz");
    el.sysinfo.textContent = JSON.stringify(ready, null, 2);
    document.querySelector('.tab[data-tab="ops"]').click();
  } catch (err) { toast(err.message, "err"); }
}

/* ============================================================ WIRING === */
function wireDrop(dropEl, inputEl, onFile, previewEl, nameEl) {
  const handle = (f) => {
    if (!f) return;
    onFile(f);
    if (nameEl) nameEl.textContent = `${f.name} — ${fmtBytes(f.size)}`;
    if (previewEl) {
      previewEl.hidden = false;
      previewEl.src = URL.createObjectURL(f);
    }
  };
  dropEl.addEventListener("click", () => inputEl.click());
  inputEl.addEventListener("change", () => handle(inputEl.files[0]));
  dropEl.addEventListener("dragover", (e) => { e.preventDefault(); dropEl.classList.add("hover"); });
  dropEl.addEventListener("dragleave", () => dropEl.classList.remove("hover"));
  dropEl.addEventListener("drop", (e) => {
    e.preventDefault(); dropEl.classList.remove("hover");
    handle(e.dataTransfer.files[0]);
  });
}

function wireUI() {
  $("btn-connect").addEventListener("click", connect);
  $("btn-info").addEventListener("click", showInfo);
  el.key.addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });

  // tabs
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
      document.querySelectorAll(".tabpanel").forEach((x) => x.classList.add("hidden"));
      t.classList.add("active");
      $("panel-" + t.dataset.tab).classList.remove("hidden");
      if (t.dataset.tab === "ops") refreshSessions();
    })
  );

  // live
  $("btn-start").addEventListener("click", startLive);
  el.btnPause.addEventListener("click", pauseLive);
  el.btnStop.addEventListener("click", stopLive);
  $("btn-refresh-voices-live").addEventListener("click", refreshVoices);
  $("btn-refresh-mics").addEventListener("click", refreshMics);

  // file tab
  wireDrop(el.dropSource, el.fileSource, (f) => { sourceFile = f; updateConvertBtn(); }, el.sourceAudio, el.sourceName);
  el.fileReference.addEventListener("change", () => {
    referenceFile = el.fileReference.files[0] || null;
    el.referenceName.textContent = referenceFile ? `${referenceFile.name} — ${fmtBytes(referenceFile.size)}` : "";
    updateConvertBtn();
  });
  document.querySelectorAll('input[name="file-target"]').forEach((r) =>
    r.addEventListener("change", () => {
      const mode = document.querySelector('input[name="file-target"]:checked').value;
      el.fileReference.disabled = mode !== "reference";
      el.fileVoice.disabled = mode === "reference";
      updateConvertBtn();
    })
  );
  $("btn-refresh-voices-file").addEventListener("click", refreshVoices);
  el.btnConvert.addEventListener("click", convertFile);

  // voices tab
  wireDrop($("drop-voice"), el.voiceFile, (f) => {
    el.btnRegisterVoice.disabled = !f;
    el.voiceFileName.textContent = `${f.name} — ${fmtBytes(f.size)}`;
    el.voiceAudio.hidden = false;
    el.voiceAudio.src = URL.createObjectURL(f);
  }, el.voiceAudio, el.voiceFileName);
  el.btnRegisterVoice.addEventListener("click", async () => {
    const f = el.voiceFile.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f, f.name);
    fd.append("name", el.voiceName.value.trim());
    el.btnRegisterVoice.disabled = true;
    try {
      const v = await api("/api/v1/voices", { method: "POST", body: fd });
      toast(`سُجّل الصوت «${v.name || v.voice_id}» ✓`, "ok");
      el.voiceName.value = ""; el.voiceFile.value = "";
      el.voiceAudio.hidden = true; el.voiceFileName.textContent = "";
      await refreshVoices();
    } catch (err) { toast(err.message, "err", 6000); }
    el.btnRegisterVoice.disabled = false;
  });
  $("btn-refresh-voices").addEventListener("click", refreshVoices);

  // ops tab
  $("btn-refresh-sessions").addEventListener("click", refreshSessions);
  $("btn-job-get").addEventListener("click", jobGet);
  $("btn-job-cancel").addEventListener("click", jobCancel);

  window.addEventListener("beforeunload", () => { try { stopLive(); } catch { /* noop */ } });
}

/* --------------------------------------------------------------- boot */
window.addEventListener("DOMContentLoaded", async () => {
  wireUI();
  el.key.value = S.apiKey;
  disableControls(false);
  await connect();
  refreshMics().catch(() => {});
  setInterval(() => {
    if (document.querySelector("#panel-ops:not(.hidden)")) refreshSessions();
  }, 5000);
});
