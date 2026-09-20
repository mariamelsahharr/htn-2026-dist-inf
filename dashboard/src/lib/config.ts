// Where the router is. Set at build time (VITE_ROUTER_URL=https://... pnpm build); empty
// means the page's own origin, which is what the dev server's proxy expects.
export const ROUTER_URL = ((import.meta.env.VITE_ROUTER_URL as string | undefined) ?? "").replace(/\/+$/, "")
