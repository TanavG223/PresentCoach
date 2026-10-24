import { McpServer } from "@modelcontextprotocol/server";
import { createMcpHandler } from "agents/mcp/server";

import { evidence } from "./evidence";
import {
  authorizationServerMetadata,
  authorize,
  bearerChallenge,
  exchangeToken,
  protectedResourceMetadata,
  registerClient,
  revokeToken,
  validateAccessToken
} from "./oauth";

type ToolPayload = Record<string, unknown>;

const jsonToolResult = (payload: ToolPayload) => ({
  content: [{ type: "text" as const, text: JSON.stringify(payload, null, 2) }],
  structuredContent: payload
});

function createServer(origin: string) {
  const server = new McpServer({
    name: "PresentCoach Inspector",
    version: "1.0.0"
  });

  server.registerTool(
    "check_presentcoach_ready",
    {
      description:
        "Return the artifact-backed PresentCoach readiness decision and its important claim boundaries.",
      inputSchema: {}
    },
    async () =>
      jsonToolResult({
        product: evidence.product.name,
        evidence_generated_at: evidence.source_generated_at,
        readiness: evidence.readiness,
        preview_url: `${origin}/`,
        privacy_boundary: evidence.privacy_boundary,
        limitations: evidence.limitations
      })
  );

  server.registerTool(
    "inspect_presentcoach_website",
    {
      description:
        "Describe the PresentCoach website workflow and return the sanitized public preview URL.",
      inputSchema: {}
    },
    async () =>
      jsonToolResult({
        product: evidence.product,
        preview_url: `${origin}/`,
        capabilities: evidence.capabilities,
        interface_workflow: [
          "Record locally or upload a presentation video.",
          "View live anonymous face-landmark tracking during a local recording.",
          "After processing, replay encrypted video beside timestamped measured coaching notes.",
          "Click a coaching note or highlighted filler word to seek to its measured moment.",
          "Compare encrypted local sessions from history."
        ],
        warning:
          "The preview is a sanitized evidence page. This connector cannot access private local sessions."
      })
  );

  server.registerTool(
    "get_presentcoach_test_evidence",
    {
      description:
        "Return the frozen real-video, tracking, and local-LLM evaluation results, including the failed holdout.",
      inputSchema: {}
    },
    async () =>
      jsonToolResult({
        product: evidence.product.name,
        evidence_generated_at: evidence.source_generated_at,
        tests: evidence.tests,
        artifacts: evidence.artifacts,
        claim_boundary:
          "These are fixed-artifact engineering and API results, not population accuracy or model training."
      })
  );

  server.registerTool(
    "get_presentcoach_privacy_boundary",
    {
      description:
        "Explain exactly what the remote inspector can and cannot access.",
      inputSchema: {}
    },
    async () =>
      jsonToolResult({
        product: evidence.product.name,
        privacy_boundary: evidence.privacy_boundary
      })
  );

  return server;
}

function escapeHtml(value: unknown): string {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

const percent = (value: number | null) =>
  value === null ? "Not evaluated" : `${value.toFixed(1)}%`;

function renderPreview(): string {
  const dev = evidence.tests.moving_presenter_development;
  const holdout = evidence.tests.one_shot_model_output_holdout;
  const status = evidence.readiness.status.replaceAll("_", " ");
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>PresentCoach Inspector</title>
  <style>
    :root{color-scheme:dark;--bg:#0d0f13;--panel:#171a21;--line:#2b303a;--ink:#f6f6f3;--muted:#aeb5c0;--green:#72e6a6;--amber:#ffcc70;--red:#ff7f87;--purple:#a78bfa}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 75% 5%,#2a2144 0,transparent 32%),var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}
    main{max-width:1160px;margin:auto;padding:56px 24px 80px}.eyebrow{font:700 12px ui-monospace,monospace;letter-spacing:.16em;color:var(--green);text-transform:uppercase}
    h1{font-size:clamp(40px,8vw,76px);line-height:.95;margin:14px 0 22px;max-width:900px}p{line-height:1.65;color:var(--muted)}.lede{font-size:19px;max-width:760px}
    .badge{display:inline-flex;gap:9px;align-items:center;border:1px solid var(--line);background:#11141a;border-radius:999px;padding:10px 14px;color:var(--green);font-weight:800}.dot{width:8px;height:8px;border-radius:50%;background:currentColor}
    .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:36px 0}.card{background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid var(--line);border-radius:20px;padding:24px;min-height:180px}.card strong{display:block;font-size:38px;margin:18px 0 6px}.pass{color:var(--green)}.fail{color:var(--red)}.warn{color:var(--amber)}
    .wide{grid-column:1/-1}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{border-left:2px solid var(--purple);padding-left:14px}.metric b{display:block;font-size:24px}.metric span{font-size:12px;color:var(--muted)}
    .privacy{border:1px solid #28503c;background:#112019;border-radius:20px;padding:24px}.privacy strong{color:var(--green)}footer{margin-top:30px;font-size:12px;color:var(--muted)}
    @media(max-width:800px){.grid,.metrics{grid-template-columns:1fr}.wide{grid-column:auto}}
  </style>
</head>
<body><main>
  <p class="eyebrow">Read-only remote MCP inspection snapshot</p>
  <h1>PresentCoach is ready—with disclosed holdout limits.</h1>
  <p class="lede">Claude can inspect this sanitized readiness page and the attached MCP tools. Private recordings, transcripts, profile data, encryption keys, and local access tokens never leave the Mac.</p>
  <span class="badge"><i class="dot"></i>${escapeHtml(status)}</span>
  <section class="grid" aria-label="Verified results">
    <article class="card"><span>REAL FLASK VIDEO PIPELINE</span><strong class="pass">${evidence.tests.real_video_api_e2e.passed}/${evidence.tests.real_video_api_e2e.case_count}</strong><p>Moving-presentation uploads passed the bounded analysis, local AI, encrypted replay, timestamp, and reload contract.</p></article>
    <article class="card"><span>LOCAL LLM GUARDRAILS</span><strong class="pass">${evidence.tests.local_llm_guardrails.passed}/${evidence.tests.local_llm_guardrails.case_count}</strong><p>Grounding, missing data, short sessions, excellent sessions, and inference bait were checked.</p></article>
    <article class="card"><span>ONE-SHOT FINAL HOLDOUT</span><strong class="fail">FAILED</strong><p>The failure is preserved without retuning: false-valid detections and dropouts exceeded their gates.</p></article>
    <article class="card wide"><p class="eyebrow">Moving-presenter tracking</p><div class="metrics">
      <div class="metric"><b>${percent(dev.trackable_coverage_percent)}</b><span>development coverage</span></div>
      <div class="metric"><b>${percent(dev.false_valid_rate_percent)}</b><span>development false-valid</span></div>
      <div class="metric"><b>${percent(holdout.trackable_coverage_percent)}</b><span>holdout coverage</span></div>
      <div class="metric"><b class="fail">${percent(holdout.false_valid_rate_percent)}</b><span>holdout false-valid</span></div>
    </div></article>
  </section>
  <section class="privacy"><strong>Privacy boundary:</strong><p>This connector is a frozen public evidence snapshot. It cannot reach localhost and contains no video, audio, transcript, identity, profile, access capability, or private session information.</p></section>
  <footer>Evidence generated ${escapeHtml(evidence.source_generated_at)} · MCP endpoint: /mcp · Local app remains at 127.0.0.1:8765 and is not publicly exposed.</footer>
</main></body></html>`;
}

const securityHeaders = {
  "Content-Security-Policy":
    "default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY"
};

export default {
  async fetch(request: Request, env: unknown, ctx: ExecutionContext) {
    const url = new URL(request.url);
    if (url.pathname === "/mcp") {
      if (request.method !== "OPTIONS" && !(await validateAccessToken(request, url.origin))) {
        return bearerChallenge(url.origin);
      }
      return createMcpHandler(() => createServer(url.origin))(request, env, ctx);
    }
    if (
      url.pathname === "/.well-known/oauth-protected-resource" ||
      url.pathname === "/.well-known/oauth-protected-resource/mcp"
    ) {
      return Response.json(protectedResourceMetadata(url.origin), {
        headers: { "Cache-Control": "public, max-age=300", ...securityHeaders }
      });
    }
    if (
      url.pathname === "/.well-known/oauth-authorization-server" ||
      url.pathname === "/.well-known/openid-configuration"
    ) {
      return Response.json(authorizationServerMetadata(url.origin), {
        headers: { "Cache-Control": "public, max-age=300", ...securityHeaders }
      });
    }
    if (url.pathname === "/register" || url.pathname === "/oauth/register") {
      return registerClient(request);
    }
    if (url.pathname === "/authorize") {
      return authorize(request, url.origin);
    }
    if (url.pathname === "/token" || url.pathname === "/oauth/token") {
      return exchangeToken(request, url.origin);
    }
    if (url.pathname === "/revoke" || url.pathname === "/oauth/revoke") {
      return revokeToken(request);
    }
    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response("Method not allowed", {
        status: 405,
        headers: { Allow: "GET, HEAD", ...securityHeaders }
      });
    }
    if (url.pathname === "/health") {
      return Response.json(
        { status: "ready", connector: "presentcoach-inspector", read_only: true },
        { headers: { "Cache-Control": "no-store", ...securityHeaders } }
      );
    }
    if (url.pathname === "/readiness.json") {
      return Response.json(evidence, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Cache-Control": "public, max-age=300",
          ...securityHeaders
        }
      });
    }
    if (url.pathname === "/") {
      return new Response(request.method === "HEAD" ? null : renderPreview(), {
        headers: {
          "Content-Type": "text/html; charset=utf-8",
          "Cache-Control": "public, max-age=300",
          ...securityHeaders
        }
      });
    }
    return new Response("Not found", { status: 404, headers: securityHeaders });
  }
} satisfies ExportedHandler;
