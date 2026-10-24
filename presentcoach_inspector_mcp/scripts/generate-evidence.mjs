import { createHash } from "node:crypto";
import { readFile, rename, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, "../..");
const outputPath = resolve(scriptDir, "../src/evidence.ts");

const sources = {
  apiE2e: "reports/presentcoach_e2e_development.json",
  development: "reports/presentcoach_dynamic_development.json",
  holdout: "reports/presentcoach_dynamic_holdout.json",
  llm: "reports/presentcoach_llm_eval.json"
};

async function load(relativePath) {
  const bytes = await readFile(resolve(repoRoot, relativePath));
  return {
    data: JSON.parse(bytes.toString("utf8")),
    sha256: createHash("sha256").update(bytes).digest("hex")
  };
}

const loaded = Object.fromEntries(
  await Promise.all(
    Object.entries(sources).map(async ([name, path]) => [name, await load(path)])
  )
);

const { apiE2e, development, holdout, llm } = Object.fromEntries(
  Object.entries(loaded).map(([name, value]) => [name, value.data])
);

function requireCondition(condition, message) {
  if (!condition) throw new Error(`Evidence contract failed: ${message}`);
}

requireCondition(
  apiE2e?.overall?.status === "passed" &&
    apiE2e.overall.case_count === 10 &&
    apiE2e.overall.passed === 10,
  "real-video API E2E report must remain 10/10 passed"
);
requireCondition(
  development?.split === "development" &&
    development.case_count === 10 &&
    development?.release_gate?.status === "pass",
  "development tracking report must remain a 10-case gate pass"
);
requireCondition(
  holdout?.split === "model_output_holdout" &&
    holdout.case_count === 5 &&
    holdout?.release_gate?.status === "fail",
  "one-shot holdout failure must stay visible"
);
requireCondition(
  llm?.case_count === 30 && llm.passed === 30 && llm.pass_rate_percent === 100,
  "local LLM guardrail report must remain 30/30 passed"
);
requireCondition(
  apiE2e.evaluation_only === true &&
    apiE2e.used_for_training === false &&
    llm.evaluation_only === true &&
    llm.used_for_training === false,
  "reports must remain evaluation-only and unused for training"
);

const sourceGeneratedAt = [
  apiE2e.generated_at,
  development.generated_at,
  holdout.generated_at,
  llm.generated_at
].sort().at(-1);

const evidence = {
  schema_version: "presentcoach.inspector.v1",
  source_generated_at: sourceGeneratedAt,
  product: {
    name: "PresentCoach",
    purpose: "A local presentation-practice coach with measured, timestamped review cues.",
    local_application_url: "http://127.0.0.1:8765/",
    remote_preview_is_sanitized_snapshot: true
  },
  readiness: {
    status: "ready_with_known_holdout_limitations",
    ready_for_local_use: true,
    complete_pipeline_verified: true,
    universal_tracking_accuracy_proven: false,
    summary:
      "The local app, encrypted replay, timestamp seeking, local LLM grounding, and development pipeline are verified. A frozen one-shot holdout exposed false-valid and dropout weaknesses that remain disclosed."
  },
  capabilities: {
    record_locally: true,
    upload_video: true,
    live_anonymous_face_landmarks: true,
    post_session_timestamped_coaching: true,
    local_whisper_transcript: true,
    local_ollama_feedback_selection: true,
    encrypted_local_history_and_replay: true,
    per_person_identity_learning: false,
    identity_recognition: false,
    live_spoken_coaching_during_recording: false
  },
  tests: {
    real_video_api_e2e: {
      status: apiE2e.overall.status,
      passed: apiE2e.overall.passed,
      case_count: apiE2e.overall.case_count,
      pass_rate_percent: apiE2e.overall.pass_rate_percent,
      scope:
        "Actual Flask upload, bounded decode, MediaPipe, Whisper, Ollama, encrypted store, signed range replay, cues, and reload on moving presentations."
    },
    local_llm_guardrails: {
      status: "passed",
      model: llm.model,
      model_digest: llm.model_digest,
      passed: llm.passed,
      case_count: llm.case_count,
      pass_rate_percent: llm.pass_rate_percent,
      scope:
        "Grounding, excellent sessions, missing quality, short-session refusal, and appearance/accent/confidence/reading/posture bait."
    },
    moving_presenter_development: {
      status: development.release_gate.status,
      case_count: development.case_count,
      trackable_coverage_percent: development.aggregate.trackable_coverage * 100,
      false_valid_rate_percent: development.aggregate.false_valid_rate * 100,
      multi_face_abstention_percent: development.aggregate.multi_face_abstention * 100,
      dropouts_per_annotated_minute:
        development.aggregate.dropouts_per_annotated_minute,
      reacquisition_success_percent:
        development.aggregate.reacquisition_success_rate * 100,
      max_reacquisition_latency_seconds:
        development.aggregate.max_reacquisition_latency_seconds
    },
    one_shot_model_output_holdout: {
      status: holdout.release_gate.status,
      case_count: holdout.case_count,
      rerun_or_retuned_after_result: false,
      failed_required_metrics: holdout.release_gate.failed_required_metrics,
      underpowered_required_metrics:
        holdout.release_gate.underpowered_required_metrics,
      trackable_coverage_percent: holdout.aggregate.trackable_coverage * 100,
      false_valid_rate_percent: holdout.aggregate.false_valid_rate * 100,
      multi_face_abstention_percent: holdout.aggregate.multi_face_abstention,
      dropouts_per_annotated_minute:
        holdout.aggregate.dropouts_per_annotated_minute,
      reacquisition_success_percent:
        holdout.aggregate.reacquisition_success_rate * 100,
      max_reacquisition_latency_seconds:
        holdout.aggregate.max_reacquisition_latency_seconds
    }
  },
  privacy_boundary: {
    connector_is_read_only: true,
    connector_contains_recordings: false,
    connector_contains_transcripts: false,
    connector_contains_profile_data: false,
    connector_contains_private_access_tokens: false,
    connector_can_reach_localhost: false,
    connector_exposes_only_sanitized_release_evidence: true
  },
  limitations: [
    "This remote inspector is a frozen sanitized evidence snapshot, not a live tunnel into the local Flask app.",
    "The holdout failed false-valid and dropout gates, so universal tracking accuracy is not claimed.",
    "Natural presentation videos lack independently labeled camera-contact gaze truth.",
    "MediaPipe and YuNet are pinned pretrained anonymous geometry models; PresentCoach does not learn identity."
  ],
  artifacts: Object.entries(sources).map(([name, path]) => ({
    name,
    path,
    sha256: loaded[name].sha256
  }))
};

const body = `// Generated by scripts/generate-evidence.mjs. Do not edit by hand.\nexport const evidence = ${JSON.stringify(evidence, null, 2)} as const;\n`;
const temporaryPath = `${outputPath}.tmp`;
await writeFile(temporaryPath, body, { encoding: "utf8", mode: 0o600 });
await rename(temporaryPath, outputPath);
