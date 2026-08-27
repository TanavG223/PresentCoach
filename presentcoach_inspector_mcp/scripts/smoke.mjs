import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

const baseUrl = process.env.PRESENTCOACH_INSPECTOR_URL ?? "http://127.0.0.1:8788";
const expectedTools = [
  "check_presentcoach_ready",
  "get_presentcoach_privacy_boundary",
  "get_presentcoach_test_evidence",
  "inspect_presentcoach_website"
];

const health = await fetch(`${baseUrl}/health`);
if (!health.ok || (await health.json()).status !== "ready") {
  throw new Error(`Health check failed: ${health.status}`);
}

const preview = await fetch(`${baseUrl}/`);
const previewText = await preview.text();
if (
  !preview.ok ||
  !previewText.includes("PresentCoach is ready") ||
  !previewText.includes("FAILED") ||
  !previewText.includes("Private recordings")
) {
  throw new Error("Sanitized preview contract failed");
}

const challengeResponse = await fetch(`${baseUrl}/mcp`);
if (
  challengeResponse.status !== 401 ||
  !challengeResponse.headers.get("www-authenticate")?.includes("oauth-protected-resource")
) {
  throw new Error("MCP endpoint did not advertise OAuth protected-resource metadata");
}

const redirectUri = "https://example.com/presentcoach-smoke-callback";
const registration = await fetch(`${baseUrl}/register`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    client_name: "PresentCoach smoke",
    redirect_uris: [redirectUri],
    token_endpoint_auth_method: "none",
    grant_types: ["authorization_code", "refresh_token"],
    response_types: ["code"]
  })
});
if (registration.status !== 201) {
  throw new Error(`OAuth client registration failed: ${registration.status}`);
}
const { client_id: clientId } = await registration.json();
const verifier = "presentcoach-smoke-verifier-with-more-than-forty-three-characters";
const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier));
const challenge = Buffer.from(digest).toString("base64url");
const authorizeUrl = new URL(`${baseUrl}/authorize`);
authorizeUrl.search = new URLSearchParams({
  response_type: "code",
  client_id: clientId,
  redirect_uri: redirectUri,
  state: "smoke-state",
  code_challenge: challenge,
  code_challenge_method: "S256",
  resource: `${baseUrl}/mcp`,
  scope: "presentcoach:read"
}).toString();
const authorization = await fetch(authorizeUrl, { redirect: "manual" });
if (authorization.status !== 302) {
  throw new Error(`OAuth authorization failed: ${authorization.status}`);
}
const authorizationRedirect = new URL(authorization.headers.get("location"));
if (authorizationRedirect.searchParams.get("state") !== "smoke-state") {
  throw new Error("OAuth state was not preserved");
}
const tokenResponse = await fetch(`${baseUrl}/token`, {
  method: "POST",
  headers: { "Content-Type": "application/x-www-form-urlencoded" },
  body: new URLSearchParams({
    grant_type: "authorization_code",
    client_id: clientId,
    code: authorizationRedirect.searchParams.get("code"),
    redirect_uri: redirectUri,
    code_verifier: verifier,
    resource: `${baseUrl}/mcp`
  })
});
if (!tokenResponse.ok) {
  throw new Error(`OAuth token exchange failed: ${tokenResponse.status}`);
}
const { access_token: accessToken } = await tokenResponse.json();

const client = new Client({ name: "presentcoach-inspector-smoke", version: "1.0.0" });
const transport = new StreamableHTTPClientTransport(new URL(`${baseUrl}/mcp`), {
  requestInit: { headers: { Authorization: `Bearer ${accessToken}` } }
});

try {
  await client.connect(transport);
  const listed = await client.listTools();
  const toolNames = listed.tools.map((tool) => tool.name).sort();
  if (JSON.stringify(toolNames) !== JSON.stringify(expectedTools)) {
    throw new Error(`Unexpected tool set: ${toolNames.join(", ")}`);
  }

  for (const name of expectedTools) {
    const result = await client.callTool({ name, arguments: {} });
    if (result.isError) throw new Error(`${name} returned an MCP error`);
    const text = result.content
      .filter((item) => item.type === "text")
      .map((item) => item.text)
      .join("\n");
    if (!text.includes("PresentCoach")) {
      throw new Error(`${name} omitted the product name`);
    }
    for (const forbidden of ["access-url", "transcript_words", "profile_id", "playback_url"]) {
      if (text.includes(forbidden)) {
        throw new Error(`${name} leaked forbidden field: ${forbidden}`);
      }
    }
  }
} finally {
  await client.close();
}

console.log(`PresentCoach Inspector MCP smoke passed: ${expectedTools.length} tools`);
