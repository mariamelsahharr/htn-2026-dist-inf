import { Check, ChevronsUpDown, Search } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { useModels } from "@/hooks/useStats"
import { colorFor, type Model } from "@/lib/api"
import { useSettings } from "@/lib/settings"

const TIER_LABEL: Record<string, string> = {
  cluster: "Raspberry Pis",
  baseten: "Baseten",
  openai: "OpenAI",
  gemini: "Gemini",
  snowflake: "Snowflake",
}
const TIER_ORDER = ["cluster", "baseten", "openai", "gemini", "snowflake"]

function group(models: Model[], query: string): [string, Model[]][] {
  const q = query.trim().toLowerCase()
  const byTier = new Map<string, Model[]>()
  for (const m of models) {
    if (q && !m.id.toLowerCase().includes(q) && !(TIER_LABEL[m.owned_by] ?? m.owned_by).toLowerCase().includes(q)) continue
    byTier.set(m.owned_by, [...(byTier.get(m.owned_by) ?? []), m])
  }
  return [...byTier.entries()].sort(
    ([a], [b]) => (TIER_ORDER.indexOf(a) + 1 || 99) - (TIER_ORDER.indexOf(b) + 1 || 99),
  )
}

// Every model the router can reach, grouped by who serves it, searchable. Picking one
// pins that provider and sends that exact model; Auto leaves the choice to the router.
export function ModelPicker() {
  const { settings, update } = useSettings()
  const models = useModels()
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState("")
  const all = models.data ?? []
  const current = all.find((m) => m.id === settings.model)
  const groups = group(all, query)
  const choose = (model: string) => {
    update({ model })
    setOpen(false)
    setQuery("")
  }

  return (
    <div className="flex flex-wrap items-center gap-3">
      <span className="text-sm">Answer with</span>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button variant="outline" role="combobox" aria-expanded={open} className="w-full justify-between sm:w-[26rem]">
            <span className="flex min-w-0 items-center gap-2">
              {current ? (
                <>
                  <span className="inline-block size-2 shrink-0 rounded-full" style={{ backgroundColor: colorFor(current.owned_by) }} />
                  <span className="truncate">{current.id}</span>
                  <span className="text-muted-foreground shrink-0 text-xs">{TIER_LABEL[current.owned_by] ?? current.owned_by}</span>
                </>
              ) : (
                <span>Auto: the router decides per request</span>
              )}
            </span>
            <ChevronsUpDown className="size-4 shrink-0 opacity-60" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[min(26rem,calc(100vw-2rem))] p-0" align="start">
          <div className="border-border flex items-center gap-2 border-b px-3 py-2">
            <Search className="text-muted-foreground size-4" />
            <input
              id="model-search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={`Search ${all.length} models`}
              className="placeholder:text-muted-foreground w-full bg-transparent text-sm outline-none"
              autoComplete="off"
            />
          </div>
          <div className="max-h-80 overflow-y-auto py-1" role="listbox" aria-label="Models">
            {!query && (
              <button
                type="button"
                role="option"
                aria-selected={settings.model === "auto"}
                onClick={() => choose("auto")}
                className="hover:bg-muted flex w-full items-center gap-2 px-3 py-2 text-left text-sm"
              >
                <Check className={`size-4 ${settings.model === "auto" ? "" : "invisible"}`} />
                Auto: the router decides per request
              </button>
            )}
            {groups.map(([tier, items]) => (
              <div key={tier}>
                <div className="text-muted-foreground flex items-center gap-2 px-3 pt-2 pb-1 text-xs">
                  <span className="inline-block size-2 rounded-full" style={{ backgroundColor: colorFor(tier) }} />
                  {TIER_LABEL[tier] ?? tier}
                  <span>{items.length}</span>
                </div>
                {items.map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    role="option"
                    aria-selected={settings.model === m.id}
                    onClick={() => choose(m.id)}
                    className="hover:bg-muted flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm"
                  >
                    <Check className={`size-4 shrink-0 ${settings.model === m.id ? "" : "invisible"}`} />
                    <span className="truncate">{m.id}</span>
                  </button>
                ))}
              </div>
            ))}
            {groups.length === 0 && <p className="text-muted-foreground px-3 py-4 text-sm">No model matches "{query}".</p>}
          </div>
        </PopoverContent>
      </Popover>
      {models.error && <span className="text-critical text-xs">could not load the model list</span>}
    </div>
  )
}
