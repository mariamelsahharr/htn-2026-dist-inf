import { ChevronsUpDown } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { useModels, useStats } from "@/hooks/useStats"
import { colorFor, tierLabel, tierRank, type Model } from "@/lib/api"
import { updateSettings, useSettings } from "@/lib/settings"

const PROVIDER: Record<string, string> = { cluster: "Raspberry Pis" }
const providerName = (tier: string) => PROVIDER[tier] ?? tierLabel(tier)

function byTier(models: Model[]): [string, Model[]][] {
  const groups = new Map<string, Model[]>()
  for (const m of models) groups.set(m.owned_by, [...(groups.get(m.owned_by) ?? []), m])
  return [...groups.entries()].sort(([a], [b]) => tierRank(a) - tierRank(b))
}

function Dot({ tier }: { tier: string }) {
  return <span className="inline-block size-2 shrink-0 rounded-full" style={{ backgroundColor: colorFor(tier) }} />
}

// Every model the router can reach, grouped by who serves it, searchable with the
// keyboard. Picking one pins that provider and sends that exact model; Auto leaves
// the choice to the router. A tier the boot probe marked down says so in its heading.
export function ModelPicker() {
  const { model } = useSettings()
  const models = useModels()
  const stats = useStats()
  const [open, setOpen] = useState(false)
  const all = models.data ?? []
  const current = all.find((m) => m.id === model)
  const stalePick = model !== "auto" && models.data != null && !current
  const choose = (id: string) => {
    updateSettings({ model: id })
    setOpen(false)
  }
  const health = (tier: string) => stats.data?.tier_health?.[tier] // absent on an older router
  const down = (tier: string) => health(tier)?.startsWith("down") ?? false

  return (
    <div className="flex flex-wrap items-center gap-3">
      <label htmlFor="model-picker" className="text-sm">
        Answer with
      </label>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            id="model-picker"
            variant="outline"
            role="combobox"
            aria-expanded={open}
            aria-haspopup="listbox"
            className="w-full justify-between sm:w-[28rem]"
          >
            <span className="flex min-w-0 items-center gap-2">
              {current ? (
                <>
                  <Dot tier={current.owned_by} />
                  <span className="truncate">{current.id}</span>
                  <span className="text-muted-foreground shrink-0 text-xs">{providerName(current.owned_by)}</span>
                </>
              ) : stalePick ? (
                <>
                  <span className="truncate">{model}</span>
                  <span className="text-warn shrink-0 text-xs">not offered right now</span>
                </>
              ) : (
                <>
                  <span>Auto</span>
                  <span className="text-muted-foreground text-xs">the router decides per request</span>
                </>
              )}
            </span>
            <ChevronsUpDown className="size-4 shrink-0 opacity-60" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[min(28rem,calc(100vw-2rem))] p-0" align="start">
          <Command loop>
            <CommandInput placeholder={`Search ${all.length} models`} aria-label="Search models" autoFocus />
            <CommandList className="max-h-80">
              <CommandEmpty>No model matches.</CommandEmpty>
              <CommandGroup heading="Router">
                <CommandItem value="auto" keywords={["automatic", "router"]} onSelect={() => choose("auto")} data-checked={model === "auto"}>
                  Auto
                  <span className="text-muted-foreground text-xs">the router decides per request</span>
                </CommandItem>
              </CommandGroup>
              {byTier(all).map(([tier, items]) => (
                <CommandGroup
                  key={tier}
                  heading={
                    <span className="flex items-center gap-2">
                      <Dot tier={tier} />
                      {providerName(tier)}
                      <span>{items.length}</span>
                      {down(tier) && <span className="text-critical normal-case">down</span>}
                    </span>
                  }
                >
                  {items.map((m) => (
                    <CommandItem key={m.id} value={m.id} keywords={[providerName(tier)]} onSelect={() => choose(m.id)} data-checked={model === m.id}>
                      <span className="truncate">{m.id}</span>
                    </CommandItem>
                  ))}
                </CommandGroup>
              ))}
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
      {models.error && <span className="text-critical text-xs">could not load the model list</span>}
    </div>
  )
}
