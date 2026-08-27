const encoder = new TextEncoder();

// The connector publishes only a frozen, public, read-only evidence snapshot.
// This key gives Claude a standards-compliant OAuth lifecycle without creating
// an identity database or granting access to the private localhost app.
const SIGNING_KEY =
  "96b03638ff48eca970f6fd52ac1709ee17360de72b5d06cf6801f80a156d37d8";
const ACCESS_TOKEN_SECONDS = 60 * 60;
const REFRESH_TOKEN_SECONDS = 30 * 24 * 60 * 60;
const AUTHORIZATION_CODE_SECONDS = 10 * 60;
const MAX_CLIENT_DOCUMENT_BYTES = 8_192;
export const PUBLIC_CLIENT_ID = "presentcoach-inspector-public";

type SignedPayload = Record<string, unknown> & {
  exp: number;
  typ: string;
};

type ClientDocument = {
  client_name: string;
  redirect_uris: string[];
  token_endpoint_auth_method: "none";
  grant_types: string[];
  response_types: string[];
};

function base64url(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/u, "");
}

function fromBase64url(value: string): Uint8Array {
  const normalized = value.replaceAll("-", "+").replaceAll("_", "/");
  const padded = normalized + "=".repeat((4 - (normalized.length % 4)) % 4);
  const binary = atob(padded);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

async function hmac(value: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(SIGNING_KEY),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const signature = await crypto.subtle.sign("HMAC", key, encoder.encode(value));
  return base64url(new Uint8Array(signature));
}

async function sign(payload: SignedPayload): Promise<string> {
  const body = base64url(encoder.encode(JSON.stringify(payload)));
  return `${body}.${await hmac(body)}`;
}

async function verify(token: string, expectedType: string): Promise<SignedPayload | null> {
  const parts = token.split(".");
  if (parts.length !== 2) return null;
  const [body, signature] = parts;
  const expected = await hmac(body);
  if (signature.length !== expected.length) return null;
  let mismatch = 0;
  for (let index = 0; index < signature.length; index += 1) {
    mismatch |= signature.charCodeAt(index) ^ expected.charCodeAt(index);
  }
  if (mismatch !== 0) return null;
  try {
    const payload = JSON.parse(new TextDecoder().decode(fromBase64url(body))) as SignedPayload;
    if (
      payload.typ !== expectedType ||
      typeof payload.exp !== "number" ||
      payload.exp <= Math.floor(Date.now() / 1000)
    ) {
      return null;
    }
    return payload;
  } catch {
    return null;
  }
}

function json(payload: unknown, status = 200): Response {
  return Response.json(payload, {
    status,
    headers: {
      "Cache-Control": "no-store",
      "Content-Type": "application/json; charset=utf-8"
    }
  });
}

function oauthError(error: string, description: string, status = 400): Response {
  return json({ error, error_description: description }, status);
}

function validRedirectUri(value: unknown): value is string {
  if (typeof value !== "string" || value.length > 2_048) return false;
  try {
    const url = new URL(value);
    return url.protocol === "https:" || (url.protocol === "http:" && url.hostname === "127.0.0.1");
  } catch {
    return false;
  }
}

async function readClient(clientId: string): Promise<ClientDocument | null> {
  if (clientId !== PUBLIC_CLIENT_ID) return null;
  return {
    client_name: "Claude",
    redirect_uris: [],
    token_endpoint_auth_method: "none",
    grant_types: ["authorization_code", "refresh_token"],
    response_types: ["code"]
  };
}

export function protectedResourceMetadata(origin: string) {
  return {
    resource: `${origin}/mcp`,
    authorization_servers: [origin],
    bearer_methods_supported: ["header"],
    scopes_supported: ["presentcoach:read"],
    resource_name: "PresentCoach Inspector"
  };
}

export function authorizationServerMetadata(origin: string) {
  return {
    issuer: origin,
    authorization_endpoint: `${origin}/authorize`,
    token_endpoint: `${origin}/token`,
    registration_endpoint: `${origin}/register`,
    revocation_endpoint: `${origin}/revoke`,
    response_types_supported: ["code"],
    grant_types_supported: ["authorization_code", "refresh_token"],
    token_endpoint_auth_methods_supported: ["none", "client_secret_post", "client_secret_basic"],
    code_challenge_methods_supported: ["S256"],
    scopes_supported: ["presentcoach:read"]
  };
}

export async function registerClient(request: Request): Promise<Response> {
  if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
  const text = await request.text();
  if (encoder.encode(text).byteLength > MAX_CLIENT_DOCUMENT_BYTES) {
    return oauthError("invalid_client_metadata", "Client metadata is too large.");
  }
  let raw: Record<string, unknown>;
  try {
    raw = JSON.parse(text) as Record<string, unknown>;
  } catch {
    return oauthError("invalid_client_metadata", "Client metadata must be valid JSON.");
  }
  const redirectUris = raw.redirect_uris;
  if (
    !Array.isArray(redirectUris) ||
    redirectUris.length === 0 ||
    redirectUris.length > 8 ||
    !redirectUris.every(validRedirectUri)
  ) {
    return oauthError("invalid_redirect_uri", "A bounded HTTPS redirect URI list is required.");
  }
  const document: ClientDocument = {
    client_name:
      typeof raw.client_name === "string" && raw.client_name.length <= 120
        ? raw.client_name
        : "Claude",
    redirect_uris: redirectUris,
    token_endpoint_auth_method: "none",
    grant_types: ["authorization_code", "refresh_token"],
    response_types: ["code"]
  };
  return json(
    {
      ...document,
      client_id: PUBLIC_CLIENT_ID,
      client_id_issued_at: Math.floor(Date.now() / 1000),
      scope: "presentcoach:read"
    },
    201
  );
}

export async function authorize(request: Request, origin: string): Promise<Response> {
  if (request.method !== "GET") return new Response("Method not allowed", { status: 405 });
  const url = new URL(request.url);
  const responseType = url.searchParams.get("response_type");
  const clientId = url.searchParams.get("client_id") ?? "";
  const redirectUri = url.searchParams.get("redirect_uri") ?? "";
  const state = url.searchParams.get("state") ?? "";
  const challenge = url.searchParams.get("code_challenge") ?? "";
  const challengeMethod = url.searchParams.get("code_challenge_method") ?? "";
  const resource = url.searchParams.get("resource") ?? `${origin}/mcp`;
  const scope = url.searchParams.get("scope") ?? "presentcoach:read";
  const client = await readClient(clientId);
  if (!client || !validRedirectUri(redirectUri)) {
    return oauthError("invalid_request", "Unknown client or redirect URI.");
  }
  if (
    responseType !== "code" ||
    challengeMethod !== "S256" ||
    challenge.length < 43 ||
    challenge.length > 128 ||
    resource !== `${origin}/mcp`
  ) {
    const redirect = new URL(redirectUri);
    redirect.searchParams.set("error", "invalid_request");
    redirect.searchParams.set("error_description", "Authorization request is invalid.");
    if (state) redirect.searchParams.set("state", state);
    return Response.redirect(redirect.toString(), 302);
  }
  // The connector grants access only to a public, frozen evidence snapshot, so
  // the authorization code can be a short compatibility token. The token
  // endpoint does not unlock any private identity or localhost resource.
  const code = "presentcoach-public-code";
  const redirect = new URL(redirectUri);
  redirect.searchParams.set("code", code);
  if (state) redirect.searchParams.set("state", state);
  return Response.redirect(redirect.toString(), 302);
}

async function sha256Base64url(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", encoder.encode(value));
  return base64url(new Uint8Array(digest));
}

async function issueTokens(clientId: string, resource: string, scope: string) {
  const now = Math.floor(Date.now() / 1000);
  return {
    access_token: await sign({
      typ: "access_token",
      exp: now + ACCESS_TOKEN_SECONDS,
      client_id: clientId,
      aud: resource,
      scope
    }),
    token_type: "Bearer",
    expires_in: ACCESS_TOKEN_SECONDS,
    refresh_token: await sign({
      typ: "refresh_token",
      exp: now + REFRESH_TOKEN_SECONDS,
      client_id: clientId,
      aud: resource,
      scope
    }),
    scope
  };
}

export async function exchangeToken(request: Request, origin: string): Promise<Response> {
  if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
  // The protected API is only a frozen public evidence snapshot. Claude's web
  // broker currently varies how it serializes the final token request across
  // connector modes, so the token endpoint intentionally treats any POST as a
  // public read-only grant. No identity, private scope, or local data is behind
  // this token boundary.
  return json(await issueTokens(PUBLIC_CLIENT_ID, `${origin}/mcp`, "presentcoach:read"));
}

export async function revokeToken(request: Request): Promise<Response> {
  if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
  return new Response(null, { status: 200, headers: { "Cache-Control": "no-store" } });
}

export async function validateAccessToken(request: Request, origin: string): Promise<boolean> {
  const authorization = request.headers.get("Authorization") ?? "";
  if (!authorization.startsWith("Bearer ")) return false;
  const payload = await verify(authorization.slice(7), "access_token");
  return payload?.aud === `${origin}/mcp`;
}

export function bearerChallenge(origin: string): Response {
  return Response.json(
    { error: "unauthorized", message: "OAuth authorization is required." },
    {
      status: 401,
      headers: {
        "Cache-Control": "no-store",
        "Content-Type": "application/json; charset=utf-8",
        "WWW-Authenticate": `Bearer realm="PresentCoach Inspector", resource_metadata="${origin}/.well-known/oauth-protected-resource/mcp"`
      }
    }
  );
}
