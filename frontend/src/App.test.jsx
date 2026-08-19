import { describe, expect, it } from "vitest";

import {
  binaryTimelinePath,
  formatPercentRatio,
  formatQualityMetric,
  isTrackingReportComplete,
  trackingVideoSource,
} from "./App.jsx";


describe("tracking evidence UI helpers", () => {
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
});
