# Dashboard

A separate site for the router: a chat box that streams through it with a badge
per answer saying who served it and why, the cluster with each Pi's temperature
and throttle state, decode/prefill tok/s per upstream with medians and p95, and
the last answers as a bar chart. Everything comes from the router's `/stats`
every two seconds; only the chosen model and theme are kept in the browser.

Vite 8, React 19, TypeScript, Tailwind 4, shadcn (radix), TanStack Query 5,
Recharts 3, the `openai` SDK for the streaming chat. pnpm.

```bash
pnpm install
pnpm dev                                        # :5173, proxies /v1 /stats to the router on :8000
pnpm lint                                       # oxlint
VITE_ROUTER_URL=https://router.example pnpm build   # dist/: static files, host them anywhere
```

`VITE_ROUTER_URL` is the router the built site talks to. Leave it unset for
`pnpm dev`. For the laptop to host it next to the router, put it in `.env.local`
(gitignored) and serve the build:

```bash
echo VITE_ROUTER_URL=http://localhost:8000 > .env.local   # or the laptop's LAN IP for phones
pnpm build && pnpm preview                                # http://<laptop>:4173
```

`dist/` is plain static files, so the same build also goes to Cloudflare Pages,
Vercel or the GoDaddy domain with `VITE_ROUTER_URL` set to the router's tunnel.

`src/lib/api.ts` holds the types for `/stats`; keep it in step with `stats()` in
`router/app.py` and `cluster/supervisor/status.example.json`.
