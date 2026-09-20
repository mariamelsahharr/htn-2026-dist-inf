import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useModels } from "@/hooks/useStats"
import { colorFor } from "@/lib/api"
import { useSettings } from "@/lib/settings"

const TIER_LABEL: Record<string, string> = {
  cluster: "the Pis",
  baseten: "Baseten",
  openai: "OpenAI",
  gemini: "Gemini",
  snowflake: "Snowflake",
}

// One choice: which model answers. Each entry names the tier that owns it, so picking a
// model is picking a provider. Auto leaves the decision to the router.
export function ModelPicker() {
  const { settings, update } = useSettings()
  const models = useModels()
  const ids = new Set(["auto", ...(models.data ?? []).map((m) => m.id)])
  return (
    <div className="flex flex-wrap items-center gap-3">
      <Label htmlFor="model">Answer with</Label>
      <Select value={ids.has(settings.model) ? settings.model : "auto"} onValueChange={(model) => update({ model })}>
        <SelectTrigger id="model" className="w-full sm:w-[26rem]">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="auto">Auto: the router decides per request</SelectItem>
          {(models.data ?? []).map((m) => (
            <SelectItem key={m.id} value={m.id}>
              <span className="flex items-center gap-2">
                <span className="inline-block size-2 rounded-full" style={{ backgroundColor: colorFor(m.owned_by) }} />
                <span>{m.id}</span>
                <span className="text-muted-foreground">on {TIER_LABEL[m.owned_by] ?? m.owned_by}</span>
              </span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
