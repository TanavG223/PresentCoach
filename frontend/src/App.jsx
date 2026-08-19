import { useEffect, useMemo, useRef, useState } from "react";

const FILLERS = new Set(["um", "uh", "like", "so"]);

function Icon({ name, size = 20 }) {
  const paths = {
    record: <circle cx="12" cy="12" r="7" fill="currentColor" stroke="none" />,
    stop: <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none" />,
    mic: <><rect x="9" y="3" width="6" height="12" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" /></>,
    scan: <><path d="M4 9V5a1 1 0 0 1 1-1h4M15 4h4a1 1 0 0 1 1 1v4M20 15v4a1 1 0 0 1-1 1h-4M9 20H5a1 1 0 0 1-1-1v-4" /><circle cx="9" cy="10" r="1" fill="currentColor" stroke="none" /><circle cx="15" cy="10" r="1" fill="currentColor" stroke="none" /><path d="M9 15c2 1.5 4 1.5 6 0" /></>,
    spark: <><path d="m12 3 1.7 5.3L19 10l-5.3 1.7L12 17l-1.7-5.3L5 10l5.3-1.7z" /><path d="m19 16 .7 2.3L22 19l-2.3.7L19 22l-.7-2.3L16 19l2.3-.7z" /></>,
    lock: <><rect x="5" y="10" width="14" height="11" rx="2" /><path d="M8 10V7a4 4 0 0 1 8 0v3" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    close: <path d="m6 6 12 12M18 6 6 18" />,
    play: <path d="m8 5 11 7-11 7z" fill="currentColor" stroke="none" />,
    upload: <><path d="M12 16V4M7 9l5-5 5 5" /><path d="M5 14v5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-5" /></>,
    chart: <><path d="M4 19V9M10 19V4M16 19v-7M22 19H2" /></>,
  };
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{paths[name]}</svg>;
}

async function api(path, options = {}, csrf = "") {
  const headers = new Headers(options.headers || {});
  if (options.method && options.method !== "GET") headers.set("X-CSRF-Token", csrf);
  if (options.body && !(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const response = await fetch(path, { ...options, headers, credentials: "same-origin" });
  const document = await response.json().catch(() => ({ error: "The local app returned an unreadable response." }));
  if (!response.ok) throw new Error(document.error || `Local request failed (${response.status})`);
  return document;
}

function formatTime(seconds = 0) {
  const whole = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(whole / 60)).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")}`;
}

function metricLabel(metric) {
  return ({ face_detected: "face framing / detection", eye_contact: "camera contact", head_stability: "head movement", expression_variety: "expression movement", audio_clear: "audio / transcript", eye_contact_percent: "Camera contact", head_rotation_std_degrees: "Head rotation variation", head_position_std_percent: "Head position variation", expression_variety_index: "Expression movement", face_presence_percent: "Face presence", overall_words_per_minute: "Overall pace", filler_count: "Tracked filler phrases", filler_rate_per_minute: "Tracked filler rate", strict_filler_count: "Um / uh", strict_filler_rate_per_minute: "Um / uh rate", pauses_over_2_seconds: "Transcript gaps over 2s", pauses_over_3_seconds: "Long transcript gaps", long_pause_rate_per_minute: "Long transcript-gap rate", longest_pause_seconds: "Longest transcript gap", longest_gaze_break_seconds: "Longest contact break", window_words_per_minute: "Pace window", strict_filler_cluster_count: "Um / uh cluster" })[metric] || metric.replaceAll("_", " ");
}

function testActual(value) {
  if (value && typeof value === "object") return Object.values(value).join(", ");
  return String(value);
}

const TRACKING_ROLE_LABELS = {
  regression: "Regression",
  development: "Development",
  holdout: "Holdout-designated",
  derived: "Derived",
};

const TRACKING_INTERVAL_LABELS = {
  no_face: "no face",
  multiple_faces: "multiple faces",
  mixed_or_invalid: "invalid / mixed",
};

export function formatPercentRatio(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "Insufficient";
}

export function formatQualityMetric(value, quality, suffix = "") {
  return quality === "good" && Number.isFinite(value)
    ? `${value}${suffix}`
    : "Insufficient";
}

function formatTrackingInterval(interval) {
  if (!interval) return "None observed";
  const duration = Number(interval.duration_seconds);
  const durationText = Number.isFinite(duration) ? `${duration.toFixed(duration < 10 ? 2 : 1)}s` : "unknown";
  return `${formatTime(interval.start_seconds)}–${formatTime(interval.end_seconds)} • ${durationText}`;
}

export function trackingVideoSource(media) {
  const url = media?.video_url;
  const window = media?.playback_window;
  const start = Number(window?.start_seconds);
  const end = Number(window?.end_seconds);
  if (media?.installed !== true || typeof url !== "string" || !url.startsWith("/api/test-media/") || !Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start) return null;
  return `${url}#t=${start.toFixed(3)},${end.toFixed(3)}`;
}

function TrackingCase({ item, active, onPlay }) {
  const repeatability = item.repeatability || {};
  const interval = item.observed_invalid_track_interval;
  const mediaSource = trackingVideoSource(item.media);
  const playback = item.media?.playback_window || {};
  const [mediaError, setMediaError] = useState(false);
  const boundPlayback = event => {
    const video = event.currentTarget;
    const start = Number(playback.start_seconds);
    const end = Number(playback.end_seconds);
    if (Number.isFinite(start) && video.currentTime < start) video.currentTime = start;
    if (Number.isFinite(end) && video.currentTime >= end) {
      video.currentTime = end;
      video.pause();
    }
  };
  return <article className="tracking-case">
    <div className="tracking-case-head">
      <span className={`tracking-role ${item.dataset_role}`}>{TRACKING_ROLE_LABELS[item.dataset_role] || item.dataset_role}</span>
      <span className={`test-result ${item.passed ? "pass" : "fail"}`}>{item.passed ? "✓ Contract passed" : "× Contract failed"}</span>
    </div>
    <h4>{item.title}</h4>
    <p>{item.purpose}</p>
    <div className="tracking-video-shell">
      {mediaSource ? active && !mediaError ? <video
        controls
        autoPlay
        playsInline
        preload="metadata"
        src={mediaSource}
        aria-label={`Tracking evaluation excerpt: ${item.title}`}
        onLoadedMetadata={event => { event.currentTarget.currentTime = Number(playback.start_seconds) || 0; }}
        onSeeking={boundPlayback}
        onTimeUpdate={boundPlayback}
        onError={() => setMediaError(true)}
      /> : mediaError ? <button type="button" className="tracking-video-missing" onClick={() => setMediaError(false)}><Icon name="lock" size={20} /><span><strong>Playback blocked by integrity check</strong><small>Restore the verified clip, then press here to retry.</small></span></button> : <button type="button" className="tracking-video-cover" onClick={() => { setMediaError(false); onPlay(); }}><Icon name="play" size={22} /><span><strong>Watch tested excerpt</strong><small>{formatTime(playback.start_seconds)}–{formatTime(playback.end_seconds)} • loads on click</small></span></button> : <div className="tracking-video-missing"><Icon name="upload" size={20} /><span><strong>Test video not installed</strong><small>Run the verified media downloader.</small></span></div>}
      <p>{item.media?.represents_exact_evaluation_input ? "Manifest-locked source excerpt; SHA-256 verified when loaded." : "Source excerpt shown; this case duplicates a crop in memory to create the two-face input."}</p>
    </div>
    <dl className="tracking-metrics">
      <div><dt>Valid single face</dt><dd>{formatPercentRatio(item.valid_face_frame_ratio)}</dd></div>
      <div><dt>Multiple-face frames</dt><dd>{formatPercentRatio(item.multi_face_frame_ratio)}</dd></div>
      <div><dt>Dropout runs</dt><dd>{item.dropout_run_count ?? "—"}</dd></div>
      <div><dt>Landmark jitter p95</dt><dd>{Number.isFinite(item.landmark_jitter_p95) ? item.landmark_jitter_p95.toFixed(4) : "Not measurable"}</dd></div>
      <div><dt>Analyzed cadence</dt><dd>{Number.isFinite(item.analyzed_fps) ? `${item.analyzed_fps.toFixed(1)} FPS` : "—"}</dd></div>
      <div><dt>Repeatability</dt><dd>{repeatability.passed ? `${repeatability.runs ?? "—"} exact runs` : "Failed"}</dd></div>
    </dl>
    <div className="tracking-interval">
      <span>Observed longest invalid-track interval</span>
      <strong>{formatTrackingInterval(interval)}</strong>
      {interval && <small>{TRACKING_INTERVAL_LABELS[interval.kind] || "invalid track"}{interval.bracketed_by_valid_track && Number.isFinite(interval.reacquisition_seconds) ? ` • reacquired after ${interval.reacquisition_seconds.toFixed(2)}s` : ""}</small>}
    </div>
    {!!item.declared_checks?.length && <details className="tracking-checks"><summary>Declared scenario checks ({item.declared_checks.length})</summary><ul>{item.declared_checks.map(check => <li key={check.id} className={check.passed ? "pass" : "fail"}><Icon name={check.passed ? "check" : "close"} size={13} /><span>{check.label}<small>{check.expected}</small></span></li>)}</ul></details>}
    <a className="source-link tracking-source" href={item.source_url} target="_blank" rel="noreferrer">{item.license} • source ↗</a>
  </article>;
}

function Onboarding({ csrf, onCreated }) {
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  async function submit(event) {
    event.preventDefault();
    try {
      await api("/api/profiles", { method: "POST", body: JSON.stringify({ name }) }, csrf);
      onCreated();
    } catch (caught) { setError(caught.message); }
  }
  return <main className="onboarding"><section className="onboarding-card">
    <div className="logo-mark"><Icon name="scan" size={30} /></div><span className="eyebrow">LOCAL • PRIVATE • OPEN SOURCE</span>
    <h1>Practice the talk.<br /><em>See the evidence.</em></h1>
    <p>PresentCoach watches your camera contact and movement, listens for pace, pauses, and filler words, then gives feedback tied to exact moments.</p>
    <form onSubmit={submit}><label>Your name<input autoFocus required maxLength="80" value={name} onChange={event => setName(event.target.value)} placeholder="Enter your name" /></label>{error && <div className="error" role="alert">{error}</div>}<button className="primary wide" type="submit">Create local workspace</button></form>
    <small><Icon name="lock" size={14} /> Encrypted on this Mac. No camera, audio, or transcript is uploaded.</small>
  </section></main>;
}

function CalibrationPanel({ calibration, csrf, profileId, onChanged }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const stage = calibration.stage;
  const step = stage === "record_baseline" ? 1 : stage === "review_baseline" ? 2 : stage === "record_repeats" || stage === "repeatability_failed" ? 3 : 4;
  async function confirm() {
    setBusy(true); setError("");
    try { await api(`/api/profiles/${encodeURIComponent(profileId)}/calibration/confirm`, { method: "POST", body: "{}" }, csrf); onChanged(); }
    catch (caught) { setError(caught.message); }
    finally { setBusy(false); }
  }
  return <section className={`calibration ${calibration.ready ? "complete" : ""}`}>
    <div className="calibration-title"><div className="step-number">{calibration.ready ? <Icon name="check" /> : step}</div><div><span className="eyebrow">OPTIONAL PERSONAL COMPARISON</span><h2>{calibration.ready ? "Baseline verified" : "Calibrate for comparison"}</h2><p>{calibration.message} General practice coaching works without this step.</p></div></div>
    <div className="step-track" aria-label={`Calibration step ${step} of 4`}>{[1, 2, 3, 4].map(value => <i key={value} className={value <= step ? "filled" : ""} />)}</div>
    {stage === "review_baseline" && <div className="baseline-review"><p>These are raw measurements—not universal targets. Look them over before using this run as your personal reference.</p><div className="baseline-values"><span><strong>{calibration.baseline?.eye_contact_percent ?? "—"}%</strong><small>camera contact</small></span><span><strong>{calibration.baseline?.overall_words_per_minute ?? "—"}</strong><small>words/min</small></span><span><strong>{calibration.baseline?.strict_filler_count ?? calibration.baseline?.filler_count ?? "—"}</strong><small>um / uh</small></span><span><strong>{calibration.baseline?.head_rotation_std_degrees ?? "—"}°</strong><small>head variation</small></span></div><button className="secondary" disabled={busy} onClick={confirm}><Icon name="check" /> I reviewed these—use as reference</button></div>}
    {calibration.repeatability && <div className="repeatability-grid">{Object.entries(calibration.repeatability).map(([key, value]) => <div key={key} className={value.passed ? "pass" : "fail"}><Icon name={value.passed ? "check" : "close"} size={15} /><span>{metricLabel(key)}</span><strong>Δ {value.delta}</strong><small>limit {value.tolerance}</small></div>)}</div>}
    {error && <div className="error" role="alert">{error}</div>}
  </section>;
}

function RecordingModal({ profileId, csrf, calibration, onClose, onSaved }) {
  const [phase, setPhase] = useState("ready");
  const [status, setStatus] = useState({ elapsed_seconds: 0, analysis_fps: 0, face_detected: false });
  const [error, setError] = useState("");
  const timer = useRef(null);
  useEffect(() => () => { clearInterval(timer.current); api("/api/recordings/cancel", { method: "POST", body: "{}" }, csrf).catch(() => {}); }, [csrf]);
  async function start() {
    setPhase("starting"); setError("");
    try {
      await api("/api/recordings/start", { method: "POST", body: JSON.stringify({ profile_id: profileId }) }, csrf); setPhase("recording");
      timer.current = setInterval(async () => { try { const update = await api("/api/recordings/status"); setStatus(update); if (update.error) setError(update.error); } catch (_) { /* stop owns transition */ } }, 500);
    } catch (caught) { setPhase("ready"); setError(caught.message); }
  }
  async function stop() {
    clearInterval(timer.current); setPhase("processing"); setError("");
    try { const result = await api("/api/recordings/stop", { method: "POST", body: "{}" }, csrf); setPhase("done"); onSaved(result); }
    catch (caught) { setPhase("ready"); setError(caught.message); }
  }
  const expected = calibration.stage === "record_baseline" ? "Baseline" : calibration.ready ? "Practice" : "Repeatability run";
  return <div className="modal-backdrop"><section className="record-modal" role="dialog" aria-modal="true" aria-labelledby="record-title">
    <button className="icon-button close" aria-label="Close recorder" onClick={onClose}><Icon name="close" /></button>
    <div className="record-head"><span className="eyebrow">{expected.toUpperCase()}</span><h2 id="record-title">{phase === "recording" ? "You’re live" : phase === "processing" ? "Analyzing locally" : "Ready when you are"}</h2><p>{phase === "processing" ? "Whisper is creating the timestamped transcript. Nothing is being uploaded." : "Speak as you normally would and look toward the camera when addressing your audience."}</p></div>
    <div className={`preview-shell ${phase}`}>{phase === "recording" ? <img src="/api/recordings/stream.mjpg" alt="Live local camera with facial landmarks" /> : <div className="preview-placeholder"><Icon name={phase === "processing" ? "spark" : "scan"} size={44} /><span>{phase === "processing" ? "Vision + audio → evidence" : "Landmarks appear here"}</span></div>}{phase === "recording" && <><div className="live-pill"><i /> REC {formatTime(status.elapsed_seconds)}</div><div className={`face-pill ${status.face_detected ? "seen" : "missing"}`}>{status.face_detected ? "Face tracked" : "Find your face"}</div><div className="fps-pill">{status.analysis_fps || 0} analysis FPS</div></>}</div>
    {phase === "recording" && <div className="live-stats"><span><Icon name="clock" /><strong>{formatTime(status.elapsed_seconds)}</strong><small>{status.elapsed_seconds < 30 ? `${Math.ceil(30 - status.elapsed_seconds)}s until feedback eligible` : "feedback duration met"}</small></span><span><Icon name="scan" /><strong>{status.analyzed_frames || 0}</strong><small>frames measured</small></span><span><Icon name="mic" /><strong>Local mic</strong><small>16 kHz • raw audio discarded</small></span></div>}
    {error && <div className="error" role="alert">{error}</div>}
    {phase === "ready" && <button className="primary record-action" onClick={start}><Icon name="record" /> Start {expected.toLowerCase()}</button>}
    {phase === "starting" && <button className="primary record-action" disabled><span className="spinner" /> Starting camera + mic…</button>}
    {phase === "recording" && <button className="stop-action" onClick={stop}><Icon name="stop" /> Stop & analyze</button>}
    {phase === "processing" && <button className="primary record-action" disabled><span className="spinner" /> Running local Whisper…</button>}
    <p className="local-note"><Icon name="lock" size={14} /> Python owns the camera and microphone. The preview is an 8 FPS local stream; metrics run at 15+ FPS.</p>
  </section></div>;
}

function UploadVideoModal({ profileId, csrf, onClose, onSaved }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [phase, setPhase] = useState("ready");
  const [error, setError] = useState("");
  useEffect(() => {
    if (!file) { setPreviewUrl(""); return undefined; }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  function choose(event) {
    const selected = event.target.files?.[0] || null;
    setError("");
    if (selected && selected.size > 512 * 1024 * 1024) {
      setFile(null); setError("Choose a video that is 512 MB or smaller."); return;
    }
    setFile(selected);
  }
  async function analyze(event) {
    event.preventDefault();
    if (!file) { setError("Choose a video first."); return; }
    setPhase("processing"); setError("");
    const body = new FormData();
    body.append("profile_id", profileId);
    body.append("video", file, file.name);
    try {
      const result = await api("/api/videos/analyze", { method: "POST", body }, csrf);
      setPhase("done"); onSaved(result);
    } catch (caught) { setPhase("ready"); setError(caught.message); }
  }
  return <div className="modal-backdrop"><section className="record-modal upload-modal" role="dialog" aria-modal="true" aria-labelledby="upload-title">
    <button className="icon-button close" aria-label="Close video importer" onClick={onClose} disabled={phase === "processing"}><Icon name="close" /></button>
    <div className="record-head"><span className="eyebrow">SAVED VIDEO • LOCAL ANALYSIS</span><h2 id="upload-title">Analyze a video file</h2><p>Use an MP4, MOV, M4V, or WebM up to 512 MB and 30 minutes. The temporary copy is deleted after analysis.</p></div>
    <form onSubmit={analyze}>
      {!previewUrl ? <label className="upload-drop"><Icon name="upload" size={34} /><strong>Choose a presentation video</strong><span>Face landmarks, audio, and timestamps are processed on this Mac.</span><input type="file" accept="video/mp4,video/quicktime,video/webm,.m4v" onChange={choose} /></label> : <div className="upload-preview"><video src={previewUrl} controls preload="metadata" /><div><strong>{file.name}</strong><span>{(file.size / (1024 * 1024)).toFixed(1)} MB • ready for local analysis</span><button type="button" className="secondary" onClick={() => setFile(null)} disabled={phase === "processing"}>Choose another</button></div></div>}
      {error && <div className="error" role="alert">{error}</div>}
      <button className="primary record-action" type="submit" disabled={!file || phase === "processing"}>{phase === "processing" ? <><span className="spinner" /> MediaPipe + Whisper are analyzing…</> : <><Icon name="spark" /> Analyze saved video</>}</button>
    </form>
    <p className="local-note"><Icon name="lock" size={14} /> Nothing is uploaded to GitHub or an external AI service. Imported videos never become calibration baselines.</p>
  </section></div>;
}

export function binaryTimelinePath(samples, x, valueFor, yesY, noY) {
  let drawing = false;
  return samples.map(sample => {
    const value = valueFor(sample);
    if (value !== true && value !== false) { drawing = false; return ""; }
    const command = drawing ? "L" : "M";
    drawing = true;
    return `${command}${x(sample.timestamp_seconds)},${value ? yesY : noY}`;
  }).filter(Boolean).join(" ");
}

function Timeline({ item }) {
  const vision = item?.metrics?.timeline?.vision || [], pace = item?.metrics?.timeline?.pace_windows || [];
  const duration = Math.max(item?.session?.duration_seconds || 1, 1), x = time => 44 + (Number(time) / duration) * 812;
  const gazePath = binaryTimelinePath(vision, x, sample => sample.eye_contact, 52, 126);
  const presencePath = binaryTimelinePath(vision, x, sample => sample.face_detected, 160, 218);
  const maxPace = Math.max(220, ...pace.map(window => Number(window.words_per_minute || 0)));
  const pacePath = pace.map((window, index) => `${index ? "L" : "M"}${x(window.start_seconds)},${218 - (Number(window.words_per_minute) / maxPace) * 166}`).join(" ");
  if (!vision.length && !pace.length) return <div className="empty-panel">No usable timeline data for this run.</div>;
  return <div className="timeline-wrap"><svg className="timeline" viewBox="0 0 900 260" role="img" aria-label="Camera contact, face presence, and speaking pace over time">{[52, 94, 136, 178, 220].map(y => <line key={y} x1="44" x2="856" y1={y} y2={y} className="grid" />)}{gazePath && <path d={gazePath} className="gaze-line" />}{presencePath && <path d={presencePath} className="presence-line" />}{pacePath && <path d={pacePath} className="pace-line" />}<text x="44" y="246">0:00</text><text x="820" y="246">{formatTime(duration)}</text></svg><div className="legend"><span className="gaze">Camera contact</span><span className="presence">Face present</span><span className="pace">Pace</span></div></div>;
}

function Transcript({ session }) {
  const words = session?.transcript || [];
  if (!words.length) return <div className="empty-panel">Audio was insufficient or no speech was detected.</div>;
  return <div className="transcript">{words.map((word, index) => { const normalized = word.text.toLowerCase().replace(/[^a-z']/g, ""); const filler = FILLERS.has(normalized) || (normalized === "you" && words[index + 1]?.text?.toLowerCase().replace(/[^a-z']/g, "") === "know"); return <span key={`${word.start_seconds}-${index}`} className={filler ? "filler" : ""} title={`${formatTime(word.start_seconds)} • ${Math.round((word.probability || 0) * 100)}% transcript confidence`}>{word.text} </span>; })}</div>;
}

function FeedbackPanel({ feedback, onGenerate }) {
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  async function generate() { setBusy(true); setError(""); try { await onGenerate(); } catch (caught) { setError(caught.message); } finally { setBusy(false); } }
  if (!feedback || feedback.status !== "ready") {
    const legacyLock = feedback?.status === "calibration_required";
    return <div className="feedback-locked"><div><Icon name={legacyLock ? "spark" : "lock"} size={24} /></div><h3>{feedback?.status === "refused_short_session" ? "Recording too short" : feedback?.status === "local_ai_unavailable" ? "Feedback failed verification" : feedback?.status === "insufficient_data" ? "Insufficient measured data" : "Feedback is ready to generate"}</h3><p>{legacyLock ? "This recording was saved under the old calibration lock. Generate an actionable, evidence-locked summary now." : feedback?.message || "Record at least 30 seconds with clear face and audio data."}</p>{legacyLock && <button className="primary generate-feedback" onClick={generate} disabled={busy}>{busy ? <><span className="spinner" /> Running local AI…</> : <><Icon name="spark" /> Generate feedback</>}</button>}{error && <div className="error" role="alert">{error}</div>}</div>;
  }
  const descriptive = feedback.source === "local_llm_verified_descriptive";
  const general = feedback.source === "local_llm_verified_general_practice";
  return <div className="feedback-columns">
    {general && <p className="feedback-mode"><Icon name="scan" size={14} /> General practice bands are active. Your completed calibration adds personal comparison.</p>}
    {descriptive && <div className="feedback-mode"><Icon name="scan" size={14} /><span>This is an older neutral summary.</span><button className="secondary upgrade-feedback" onClick={generate} disabled={busy}>{busy ? "Upgrading…" : "Upgrade coaching"}</button></div>}
    {error && <div className="error" role="alert">{error}</div>}
    <div><span className="feedback-label strength"><Icon name="check" size={15} /> {descriptive ? "Measured observations" : "What worked"}</span>{feedback.strengths.map((claim, index) => <article key={index}><strong>{metricLabel(claim.metric)}</strong><p>{claim.text}</p><small>{formatTime(claim.timestamp_seconds)} • {claim.value} {claim.unit}</small></article>)}</div>
    <div><span className="feedback-label improve"><Icon name="spark" size={15} /> {descriptive ? "Notable moments" : "Practice next"}</span>{feedback.improvements.length ? feedback.improvements.map((claim, index) => <article key={index}><strong>{metricLabel(claim.metric)}</strong><p>{claim.text}</p><small>{formatTime(claim.timestamp_seconds)} • {claim.value} {claim.unit}</small></article>) : <p className="no-claims">No measured behavior crossed a practice threshold in this run.</p>}</div>
    {!!feedback.insufficient_data?.length && <p className="insufficient">Insufficient data: {feedback.insufficient_data.map(metricLabel).join(", ")}. Adjust framing/audio and try again; PresentCoach will not guess.</p>}
    <p className="feedback-boundary"><Icon name="lock" size={13} /> Reports measured delivery behavior only. It does not infer confidence or prove you were reading. Full-body posture is not measured yet.</p>
  </div>;
}

function SessionView({ item, onGenerateFeedback }) {
  if (!item) return <section className="empty-home"><div><Icon name="chart" size={32} /></div><h2>Your evidence will appear here</h2><p>Record a baseline to see raw camera contact, movement, pace, pauses, fillers, and the timestamped transcript.</p></section>;
  const aggregate = item.metrics.aggregate, quality = item.metrics.quality_flags || item.session.quality_flags;
  return <><section className="metric-row"><div><span>Camera contact</span><strong>{quality.eye_contact === "good" ? `${aggregate.eye_contact_percent}%` : "Insufficient"}</strong><small>{quality.eye_contact === "good" ? "of analyzed contact frames" : "record a new 15 FPS run"}</small></div><div><span>Speaking pace</span><strong>{quality.audio_clear === "good" ? aggregate.overall_words_per_minute : "Insufficient"}</strong><small>{quality.audio_clear === "good" ? "words per minute" : "audio quality"}</small></div><div><span>Um / uh</span><strong>{quality.audio_clear === "good" ? aggregate.strict_filler_count ?? aggregate.filler_count : "—"}</strong><small>{quality.audio_clear === "good" ? `${aggregate.strict_filler_rate_per_minute ?? "—"} per minute` : "audio quality"}</small></div><div><span>Face presence</span><strong>{quality.face_detected === "good" ? `${aggregate.face_presence_percent}%` : "Insufficient"}</strong><small>{quality.face_detected === "good" ? `${aggregate.analyzed_vision_fps} analysis FPS` : "framing or legacy precision"}</small></div></section>
    <section className="content-grid"><article className="panel timeline-panel"><div className="panel-head"><div><span className="eyebrow">MOMENT BY MOMENT</span><h2>Metric timeline</h2></div><span className="duration"><Icon name="clock" size={16} /> {formatTime(item.session.duration_seconds)}</span></div><Timeline item={item} /><div className="submetrics"><span><small>Head rotation variation</small><strong>{formatQualityMetric(aggregate.head_rotation_std_degrees, quality.head_stability, "°")}</strong></span><span><small>Position variation</small><strong>{formatQualityMetric(aggregate.head_position_std_percent, quality.head_stability, "% frame")}</strong></span><span><small>Expression movement</small><strong>{formatQualityMetric(aggregate.expression_variety_index, quality.expression_variety)}</strong></span><span><small>Pauses over 2s</small><strong>{aggregate.pauses_over_2_seconds}</strong></span></div></article>
    <article className="panel feedback-panel"><div className="panel-head"><div><span className="eyebrow">LOCAL AI • EVIDENCE-LOCKED</span><h2>Coaching notes</h2></div><Icon name="spark" /></div><FeedbackPanel feedback={item.feedback} onGenerate={() => onGenerateFeedback(item.session.session_id)} /></article>
    <article className="panel transcript-panel"><div className="panel-head"><div><span className="eyebrow">WHISPER • LOCAL</span><h2>Timestamped transcript</h2></div><span className="filler-key">highlight = tracked phrase</span></div><Transcript session={item.session} /></article>
    <article className="panel moments-panel"><div className="panel-head"><div><span className="eyebrow">NOTABLE MOMENTS</span><h2>Evidence markers</h2></div></div><ul><li><span>Longest camera-contact break</span><strong>{item.metrics.timeline.longest_gaze_break ? `${item.metrics.timeline.longest_gaze_break.duration_seconds}s at ${formatTime(item.metrics.timeline.longest_gaze_break.start_seconds)}` : "None measured"}</strong></li><li><span>Um / uh clusters</span><strong>{item.metrics.timeline.filler_clusters.length}</strong></li><li><span>Pace spikes</span><strong>{item.metrics.timeline.pace_spikes.length}</strong></li><li><span>Longest transcript gap</span><strong>{item.metrics.timeline.longest_pause ? `${item.metrics.timeline.longest_pause.duration_seconds}s at ${formatTime(item.metrics.timeline.longest_pause.start_seconds)}` : `${aggregate.longest_pause_seconds}s`}</strong></li></ul></article></section></>;
}

export function isTrackingReportComplete(tracking = {}) {
  return tracking.trusted === true
    && tracking.manifest_verified === true
    && tracking.model_verified === true
    && Number(tracking.repeatability_runs_per_case) >= 2
    && tracking.case_count > 0
    && tracking.passed === tracking.case_count
    && !tracking.partial_run;
}

function TestLab({ lab }) {
  const [activeClip, setActiveClip] = useState(null);
  if (!lab) return null;
  const video = lab.video_eval || {}, llm = lab.llm_eval || {}, tracking = lab.tracking_eval || {};
  const trackingCases = tracking.cases || [];
  const trackingComplete = isTrackingReportComplete(tracking);
  return <section className="test-lab" id="test-lab">
    <div className="test-lab-head"><div><span className="eyebrow">REPRODUCIBLE TEST LAB</span><h2>Watch the inputs. Inspect the evidence.</h2><p>Licensed clips evaluate MediaPipe, Whisper, quality flags, guardrails, and anonymous landmark continuity. These are scenario-contract results—not face-tracking accuracy—and they never train on a speaker’s face.</p></div><div className="test-summaries"><span className={video.passed === video.case_count && video.case_count ? "pass" : "fail"}><strong>{video.passed}/{video.case_count}</strong><small>full-pipeline video cases</small></span><span className={trackingComplete ? "pass" : "fail"}><strong>{tracking.trusted ? `${tracking.passed}/${tracking.case_count}` : "—"}</strong><small>{tracking.trusted ? "tracking scenario contracts" : "tracking report unverified"}</small></span><span className={llm.passed === llm.case_count && llm.case_count ? "pass" : "fail"}><strong>{llm.passed}/{llm.case_count}</strong><small>LLM cases • {llm.pass_rate_percent}%</small></span></div></div>
    <div className="test-video-grid">{lab.clips.map(clip => {
      const result = clip.result, measured = result?.measurements;
      return <article className="test-video-card" key={clip.id}>
        {clip.available ? activeClip === clip.id ? <video controls autoPlay preload="metadata" src={clip.video_url} aria-label={`Evaluation video: ${clip.title}`} /> : <button className="test-video-cover" onClick={() => setActiveClip(clip.id)}><Icon name="play" size={28} /><strong>Play test clip</strong><small>Loads only when you press play</small></button> : <div className="test-video-missing"><Icon name="upload" size={30} /><strong>Clip not installed</strong><code>zsh scripts/download_test_videos.sh</code></div>}
        <div className="test-card-body"><div className="test-card-title"><div><span className="eyebrow">{clip.license}</span><h3>{clip.title}</h3></div><span className={`test-result ${result?.passed ? "pass" : "fail"}`}>{result ? `${result.passed ? "✓" : "×"} ${result.passed ? "Passed" : "Failed"}` : "Not run"}</span></div><p>{clip.purpose}</p>
          {measured && <div className="test-measures"><span><strong>{measured.analyzed_vision_fps}</strong><small>sampled media FPS</small></span><span><strong>{measured.face_presence_percent}%</strong><small>face present</small></span><span><strong>{measured.transcript_word_count}</strong><small>timestamped words</small></span><span><strong>{formatTime(measured.duration_seconds)}</strong><small>duration</small></span></div>}
          {result?.checks && <ul className="test-checks">{result.checks.map(check => <li key={check.label} className={check.passed ? "pass" : "fail"}><Icon name={check.passed ? "check" : "close"} size={14} /><span><strong>{check.label}</strong><small>{testActual(check.actual)} • expected {check.expected}</small></span></li>)}</ul>}
          <a className="source-link" href={clip.source_url} target="_blank" rel="noreferrer">Source &amp; license ↗</a>
        </div>
      </article>;
    })}</div>
    <section className="tracking-suite" aria-labelledby="tracking-suite-title">
      <div className="tracking-suite-head">
        <div><span className="eyebrow">ANONYMOUS LANDMARK TRACKING</span><h3 id="tracking-suite-title">Face-pattern reliability, without face recognition</h3><p>These cases measure whether a single anonymous landmark track stays valid, abstains around multiple faces, survives dropouts, and repeats deterministically. No identity labels, face embeddings, or model training are involved.</p></div>
        <div className="tracking-contract" aria-label="Tracking evaluation privacy contract"><Icon name="lock" size={17} /><span><strong>Geometry only</strong><small>No identity • no embeddings • no training</small></span></div>
      </div>
      <div className="tracking-role-guide" aria-label="Dataset role legend">
        <span><i className="regression" />Regression <small>pinned behavior</small></span>
        <span><i className="development" />Development <small>used while setting checks</small></span>
        <span><i className="holdout" />Holdout-designated <small>declared reserved role</small></span>
        <span><i className="derived" />Derived <small>licensed in-memory transform</small></span>
      </div>
      {tracking.trusted && trackingCases.length ? <div className="tracking-grid" role="list" aria-label={`${tracking.passed} of ${tracking.case_count} anonymous tracking scenario contracts passed`}>
        {trackingCases.map(item => <div role="listitem" key={item.id}><TrackingCase item={item} active={activeClip === `tracking:${item.id}`} onPlay={() => setActiveClip(`tracking:${item.id}`)} /></div>)}
      </div> : <div className="tracking-empty" role="status">{tracking.trust_message || "Tracking benchmark evidence is unavailable. Run the complete local tracking evaluation to populate verified cases."}</div>}
      <p className="tracking-method">Valid and multiple-face values are frame ratios. Jitter is normalized rigid-landmark motion at the 95th percentile and appears only when a stable run exists. Repeatability compares deterministic aggregate tracking metrics and declared checks; hardware timing is excluded. Passing means these declared scenarios behaved as expected, not that general face-tracking accuracy is proven.</p>
    </section>
    <div className="test-note"><Icon name="lock" size={17} /><span><strong>Evaluation improves reliability; it does not retrain MediaPipe.</strong> A failed case means the measurement code, thresholds, or quality guardrails need work. Speakers are never enrolled or memorized.</span></div>
  </section>;
}

export default function App() {
  const [data, setData] = useState(null), [error, setError] = useState(""), [recordOpen, setRecordOpen] = useState(false), [uploadOpen, setUploadOpen] = useState(false), [selectedId, setSelectedId] = useState(null), [toast, setToast] = useState("");
  const csrf = data?.csrf_token || "";
  async function load(profileId = "") { try { const fresh = await api(`/api/bootstrap${profileId ? `?profile=${encodeURIComponent(profileId)}` : ""}`); setData(fresh); setError(""); if (!selectedId && fresh.sessions?.length) setSelectedId(fresh.sessions.at(-1).session.session_id); } catch (caught) { setError(caught.message); } }
  useEffect(() => { load(); }, []);
  const sessions = useMemo(() => data?.sessions || [], [data]), selected = sessions.find(item => item.session.session_id === selectedId) || sessions.at(-1);
  if (!data && error) return <main className="fatal"><div className="logo-mark"><Icon name="scan" /></div><h1>PresentCoach couldn’t start</h1><p>{error}</p><button className="primary" onClick={() => load()}>Try again</button></main>;
  if (!data) return <main className="loading"><span className="spinner dark" /><p>Opening the local coach…</p></main>;
  if (!data.profile) return <Onboarding csrf={csrf} onCreated={() => load()} />;
  return <div className="app-shell"><header className="topbar"><a className="brand" href="/"><span className="logo-mark small"><Icon name="scan" /></span><span><strong>PresentCoach</strong><small>evidence-based practice</small></span></a><div className="status-row"><span className={data.whisper.available ? "status ready" : "status"}><i /> Whisper local</span><span className={data.local_model.available ? "status ready" : "status"}><i /> AI {data.local_model.available ? "ready" : "offline"}</span><a className="secondary top-action test-link" href="#test-lab"><Icon name="chart" /> Tests</a><button className="secondary top-action" onClick={() => setUploadOpen(true)}><Icon name="upload" /> Upload video</button><button className="primary top-action" onClick={() => setRecordOpen(true)} disabled={data.calibration.stage === "review_baseline"}><Icon name="record" /> {data.calibration.ready ? "New practice" : data.calibration.stage === "record_baseline" ? "Record baseline" : "Record repeat"}</button></div></header>
    <main className="dashboard"><section className="welcome"><div><span className="eyebrow">PRESENTATION LAB</span><h1>Welcome back, {data.profile.name}.</h1><p>Every observation below is tied to a measured number and a moment in your recording.</p></div><div className="privacy"><Icon name="lock" /><span><strong>Runs entirely on this Mac</strong><small>Python • MediaPipe • whisper.cpp • Ollama</small></span></div></section><CalibrationPanel calibration={data.calibration} csrf={csrf} profileId={data.profile.id} onChanged={() => load(data.profile.id)} /><SessionView item={selected} onGenerateFeedback={async sessionId => { await api(`/api/profiles/${encodeURIComponent(data.profile.id)}/sessions/${encodeURIComponent(sessionId)}/feedback`, { method: "POST", body: "{}" }, csrf); await load(data.profile.id); setToast("Local evidence summary generated."); setTimeout(() => setToast(""), 4000); }} />
    <TestLab lab={data.test_lab} />
    {!!sessions.length && <section className="history"><div className="section-head"><div><span className="eyebrow">SESSION HISTORY</span><h2>Compare your runs</h2></div><strong>{sessions.length} saved locally</strong></div><div className="history-list">{[...sessions].reverse().map(item => <button key={item.session.session_id} className={selected?.session.session_id === item.session.session_id ? "selected" : ""} onClick={() => setSelectedId(item.session.session_id)}><span className={`kind ${item.session.session_kind}`}>{item.session.session_kind}</span><strong>{new Date(item.session.start_time).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</strong><small>{new Date(item.session.start_time).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })} • {formatTime(item.session.duration_seconds)}</small><div><span>{item.session.quality_flags?.eye_contact === "good" ? `${item.metrics.aggregate.eye_contact_percent}% contact` : "Contact insufficient"}</span><span>{item.metrics.aggregate.overall_words_per_minute} WPM</span><span>{item.metrics.aggregate.strict_filler_count ?? item.metrics.aggregate.filler_count} um/uh</span></div><Icon name="play" size={15} /></button>)}</div></section>}<footer><span><Icon name="lock" size={14} /> Encrypted session history</span><span>Evidence-based coaching—never a score or judgment of you.</span></footer></main>
    {recordOpen && <RecordingModal profileId={data.profile.id} csrf={csrf} calibration={data.calibration} onClose={() => setRecordOpen(false)} onSaved={result => { setRecordOpen(false); setSelectedId(result.session.session_id); setToast("Session analyzed and encrypted locally."); load(data.profile.id); setTimeout(() => setToast(""), 4000); }} />}
    {uploadOpen && <UploadVideoModal profileId={data.profile.id} csrf={csrf} onClose={() => setUploadOpen(false)} onSaved={result => { setUploadOpen(false); setSelectedId(result.session.session_id); setToast("Saved video analyzed and encrypted locally."); load(data.profile.id); setTimeout(() => setToast(""), 4000); }} />}
    {toast && <div className="toast" role="status"><Icon name="check" /> {toast}</div>}</div>;
}
