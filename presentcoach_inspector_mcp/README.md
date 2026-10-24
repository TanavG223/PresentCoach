# PresentCoach Inspector MCP

Read-only remote MCP connector for Claude.ai. It exposes a sanitized snapshot
of PresentCoach release evidence and a public inspection page. It does not
connect to the private local Flask process and includes no recordings,
transcripts, profile data, keys, or access tokens.

## Tools

- `check_presentcoach_ready`
- `inspect_presentcoach_website`
- `get_presentcoach_test_evidence`
- `get_presentcoach_privacy_boundary`

## Local verification

```bash
npm install
npm run check
npm run dev
```

The preview is served at `http://127.0.0.1:8788/` and the MCP endpoint at
`http://127.0.0.1:8788/mcp`.

## Deployment

For a persistent Claude.ai connector, authenticate Wrangler and run:

```bash
npx wrangler login
npm run deploy
```

Then add the resulting `https://…workers.dev/mcp` URL in Claude.ai under
Settings → Connectors → Add custom connector.
