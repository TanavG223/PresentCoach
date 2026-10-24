import { describe, expect, it } from "vitest";

import {
  activeCoachingCueIndex,
  activeCoachingCueIds,
  binaryTimelinePath,
  dynamicSplitPresentation,
  formatPercentRatio,
  formatQualityMetric,
  findRecoveredSession,
  groupCoachingMarkers,
  isTrackingReportComplete,
  normalizeCoachingCues,
  rankCoachingCues,
  recorderCanClose,
  requestNativeCapturePermissions,
  savedSessionNotice,
  sessionVideoSource,
  shouldWarnBeforeRecorderExit,
  sessionHistoryKind,
  trackingOutcomeLabel,
  trackingVideoSource,
  transcriptVtt,
} from "./App.jsx";


describe("tracking evidence UI helpers", () => {
  it("distinguishes an untouched holdout from failure and underpowered evidence", () => {
    expect(dynamicSplitPresentation({ state: "untouched" }, "model_output_holdout")).toEqual({
      label: "Untouched • not run",
      tone: "untouched",
    });
    expect(dynamicSplitPresentation({ state: "not_evaluated", trusted: true }, "development")).toEqual({
      label: "Not evaluated",
      tone: "neutral",
    });
    expect(dynamicSplitPresentation({ state: "pass", trusted: true, release_gate_status: "pass" })).toEqual({
      label: "Engineering gates passed",
      tone: "pass",
    });
  });

  it("renders ratios without turning missing data into zero", () => {
    expect(formatPercentRatio(0.73444)).toBe("73.4%");
    expect(formatPercentRatio(null)).toBe("Insufficient");
  });

  it("does not display zero-valued pose or expression as measured when quality is bad", () => {
    expect(formatQualityMetric(0, "bad", "°")).toBe("Insufficient");
    expect(formatQualityMetric(0, "bad")).toBe("Insufficient");
    expect(formatQualityMetric(0, "good", "°")).toBe("0°");
  });

  it("breaks the camera-contact path across ineligible gaze samples", () => {
    const samples = [
      { timestamp_seconds: 0, eye_contact: true },
      { timestamp_seconds: 1, eye_contact: null },
      { timestamp_seconds: 2, eye_contact: false },
    ];
    expect(
      binaryTimelinePath(
        samples,
        value => value * 10,
        sample => sample.eye_contact,
        5,
        15,
      ),
    ).toBe("M0,5 M20,15");
  });

  it("only marks a complete, repeated, manifest-verified suite green", () => {
    const complete = {
      trusted: true,
      manifest_verified: true,
      model_verified: true,
      repeatability_runs_per_case: 2,
      case_count: 12,
      passed: 12,
      partial_run: false,
    };
    expect(isTrackingReportComplete(complete)).toBe(true);
    expect(isTrackingReportComplete({ ...complete, passed: 11 })).toBe(false);
    expect(isTrackingReportComplete({ ...complete, repeatability_runs_per_case: 1 })).toBe(false);
    expect(isTrackingReportComplete({ ...complete, manifest_verified: false })).toBe(false);
  });

  it("builds a bounded local media fragment only from verified playback metadata", () => {
    const media = {
      installed: true,
      video_url: "/api/test-media/example/video",
      playback_window: { start_seconds: 20, end_seconds: 80 },
    };
    expect(trackingVideoSource(media)).toBe(
      "/api/test-media/example/video#t=20.000,80.000",
    );
    expect(trackingVideoSource({ ...media, video_url: "https://example.com/video" })).toBeNull();
    expect(trackingVideoSource({ ...media, playback_window: { start_seconds: 20, end_seconds: 10 } })).toBeNull();
  });

  it("accepts only signed same-origin session replay links", () => {
    const media = {
      available: true,
      playback_url: "/api/profiles/profile-1/sessions/session-1/video?token=signed-value",
    };
    expect(sessionVideoSource(media)).toBe(media.playback_url);
    expect(sessionVideoSource({ ...media, playback_url: "https://example.com/video?token=x" })).toBeNull();
    expect(sessionVideoSource({ ...media, playback_url: "/api/profiles/p/sessions/s/video" })).toBeNull();
    expect(sessionVideoSource({ ...media, available: false })).toBeNull();
  });

  it("does not claim a replay was encrypted when retention failed", () => {
    expect(savedSessionNotice({ media: { available: true } })).toBe(
      "Session analyzed and encrypted locally.",
    );
    expect(savedSessionNotice({ media: { available: false } }, "Saved video")).toBe(
      "Saved video analyzed; measurements were saved, but the video replay was not retained.",
    );
    expect(savedSessionNotice({ media: { available: false, message: "Measurements saved; replay unavailable." } })).toBe(
      "Measurements saved; replay unavailable.",
    );
  });

  it("cannot close the recorder while start or stop processing owns its resources", () => {
    expect(recorderCanClose("ready")).toBe(true);
    expect(recorderCanClose("recording")).toBe(false);
    expect(recorderCanClose("starting")).toBe(false);
    expect(recorderCanClose("processing")).toBe(false);
  });

  it("warns before ordinary navigation only while this modal owns an active take", () => {
    expect(shouldWarnBeforeRecorderExit(true)).toBe(true);
    expect(shouldWarnBeforeRecorderExit(false)).toBe(false);
    expect(shouldWarnBeforeRecorderExit(undefined)).toBe(false);
  });

  it("waits for native capture permission before starting Python devices", async () => {
    const listeners = new Map();
    const host = {
      webkit: { messageHandlers: { presentCoachPermissions: {
        postMessage(value) {
          expect(value).toBe("request");
          queueMicrotask(() => listeners.get("presentcoach-permission-result")({
            detail: { camera: true, microphone: false },
          }));
        },
      } } },
      addEventListener(name, listener) { listeners.set(name, listener); },
      removeEventListener(name) { listeners.delete(name); },
    };
    await expect(requestNativeCapturePermissions(host, 100)).resolves.toEqual({
      camera: true,
      microphone: false,
      native: true,
    });
    expect(listeners.size).toBe(0);
  });

  it("keeps browser development compatible when no native permission bridge exists", async () => {
    await expect(requestNativeCapturePermissions({}, 100)).resolves.toEqual({
      camera: true,
      microphone: true,
      native: false,
    });
  });

  it("reports stop recovery only when encrypted history contains a new durable session", () => {
    const original = { session: { session_id: "session-before" } };
    const saved = { session: { session_id: "session-after" }, media: { available: true } };
    expect(findRecoveredSession(["session-before"], [original])).toBeNull();
    expect(findRecoveredSession(["session-before"], [original, saved])).toBe(saved);
    expect(findRecoveredSession([], [])).toBeNull();
  });

  it("normalizes the evidence-locked review cue document", () => {
    const item = {
      session: { session_id: "session-1", duration_seconds: 60 },
      coaching_cues: {
        cues: [
          { cue_id: "cue-2", kind: "strict_filler", role: "improvement", start_seconds: 20, end_seconds: 20.4, seek_seconds: 20, title: "Filler detected", text: "Whisper detected um.", metric: "strict_filler_count", value: 1, unit: "um/uh filler" },
          { cue_id: "cue-1", kind: "camera_contact_break", role: "review", start_seconds: 5, end_seconds: 12, seek_seconds: 5, title: "Camera-contact break", text: "A 7 second break was measured.", metric: "longest_gaze_break_seconds", value: 7, unit: "seconds" },
          { cue_id: "cue-0", kind: "quality", role: "insufficient", start_seconds: 0, end_seconds: 60, seek_seconds: 0, title: "Audio insufficient", text: "Audio was unavailable.", metric: "audio_clear", value: null, unit: null },
        ],
      },
    };
    const cues = normalizeCoachingCues(item);
    expect(cues.map(cue => cue.id)).toEqual(["cue-0", "cue-1", "cue-2"]);
    expect(cues.map(cue => cue.category)).toEqual(["quality", "camera", "speech"]);
    expect(cues[0].end_seconds).toBe(2.5);
    expect(cues[0].seekable).toBe(false);
    expect(cues[1].seekable).toBe(true);
    expect(cues.map(cue => cue.detail).join(" ").toLowerCase()).not.toContain("not confident");
  });

  it("categorizes verified claims by their measured metric", () => {
    const item = {
      session: { duration_seconds: 60 },
      coaching_cues: { cues: [
        { cue_id: "a", kind: "verified_coaching", metric: "eye_contact_percent", role: "strength", start_seconds: 60, seekable: false },
        { cue_id: "b", kind: "verified_coaching", metric: "strict_filler_rate_per_minute", role: "improvement", start_seconds: 60, seekable: false },
        { cue_id: "c", kind: "verified_coaching", metric: "window_words_per_minute", role: "improvement", start_seconds: 15, seekable: true },
      ] },
    };
    expect(normalizeCoachingCues(item).map(cue => cue.category)).toEqual(["pace", "camera", "speech"]);
  });

  it("groups overlapping replay markers and never activates session-wide notes", () => {
    const cues = [
      { id: "summary", start_seconds: 0, end_seconds: 60, seekable: false },
      { id: "a", start_seconds: 10, end_seconds: 11, seekable: true },
      { id: "b", start_seconds: 10.3, end_seconds: 12, seekable: true },
    ];
    expect(groupCoachingMarkers(cues)).toHaveLength(1);
    expect(groupCoachingMarkers(cues)[0].cues).toHaveLength(2);
    expect([...activeCoachingCueIds(cues, 10.5)]).toEqual(["a", "b"]);
    expect([...activeCoachingCueIds(cues, 0)]).not.toContain("summary");
  });

  it("creates local WebVTT captions from timestamped transcript words", () => {
    const vtt = transcriptVtt({ transcript: [
      { text: "Hello", start_seconds: 0, end_seconds: 0.4 },
      { text: "there", start_seconds: 0.5, end_seconds: 1.0 },
    ] });
    expect(vtt).toContain("WEBVTT");
    expect(vtt).toContain("00:00:00.000 --> 00:00:01.000");
    expect(vtt).toContain("Hello there");
  });

  it("activates a cue only around its measured time range", () => {
    const cues = [
      { start_seconds: 5, end_seconds: 8 },
      { start_seconds: 20, end_seconds: 20.2 },
    ];
    expect(activeCoachingCueIndex(cues, 6)).toBe(0);
    expect(activeCoachingCueIndex(cues, 15)).toBe(-1);
    expect(activeCoachingCueIndex(cues, 22)).toBe(1);
    expect(activeCoachingCueIndex(cues, 24)).toBe(-1);
  });

  it("shows five diverse priority notes before the full measured list", () => {
    const cues = [
      ...Array.from({ length: 6 }, (_, index) => ({ id: `f${index}`, category: "speech", role: "review", severity: "info", start_seconds: index, end_seconds: index + 0.2 })),
      { id: "camera", category: "camera", role: "improvement", severity: "high", start_seconds: 20, end_seconds: 27 },
      { id: "quality", category: "quality", role: "insufficient", severity: "info", start_seconds: 0, end_seconds: 2 },
    ];
    const ranked = rankCoachingCues(cues, 5);
    expect(ranked).toHaveLength(5);
    expect(ranked.map(cue => cue.id)).toContain("camera");
    expect(ranked.map(cue => cue.id)).toContain("quality");
    expect(ranked.map(cue => cue.category)).not.toEqual(["speech", "speech", "speech", "speech", "speech"]);
  });

  it("labels only the selected quality-approved run as the baseline", () => {
    const base = {
      session: { session_id: "good", session_kind: "baseline", duration_seconds: 40 },
      metrics: { quality_flags: { face_detected: "good", eye_contact: "good", head_stability: "good", expression_variety: "good", audio_clear: "good" } },
    };
    expect(sessionHistoryKind(base, { baseline_session_id: "good" }).label).toBe("Baseline");
    expect(sessionHistoryKind({ ...base, session: { ...base.session, session_id: "old" } }, { baseline_session_id: "good" }).label).toBe("Baseline attempt");
    expect(sessionHistoryKind({ ...base, session: { ...base.session, duration_seconds: 5 } }, { baseline_session_id: "good" }).label).toBe("Baseline attempt");
  });

  it("explains a successful zero-face scenario as expected abstention", () => {
    expect(trackingOutcomeLabel({ passed: true, title: "Distant Face Abstention", valid_face_frame_ratio: 0 })).toBe("✓ Expected abstention");
    expect(trackingOutcomeLabel({ passed: true, title: "Stable presenter" })).toBe("✓ Contract passed");
    expect(trackingOutcomeLabel({ passed: false, title: "Distant Face Abstention" })).toBe("× Contract failed");
  });
});
