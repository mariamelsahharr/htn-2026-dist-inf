# Dashboard

The live view the router serves at `/`: a chat box that streams through the router
with a badge per answer saying who served it and why, the cluster with each Pi's
temperature and throttle state, decode/prefill tok/s per upstream with p50/p95,
and the last answers as a bar chart. Everything comes from the router's `/stats`
every two seconds; nothing is stored in the browser.

Vite 8, React 19, TypeScript, Tailwind 4, shadcn (radix), TanStack Query 5,
Recharts 3, the `openai` SDK for the streaming chat. pnpm.

```bash
pnpm install
pnpm dev            # :5173, proxies /v1 /stats /healthz to the router on :8000
pnpm lint           # oxlint
pnpm build          # tsc + vite -> dist/, which the router serves at /
```

`src/lib/api.ts` holds the types for `/stats`; keep it in step with `stats()` in
`router/app.py` and `cluster/supervisor/status.example.json`.
