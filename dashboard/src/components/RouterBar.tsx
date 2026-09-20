import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useModels } from "@/hooks/useStats"
import { PROVIDERS, useSettings } from "@/lib/settings"

// Where the page points and how requests are shaped. Changes apply to the next message.
export function RouterBar() {
  const { settings, update } = useSettings()
  const models = useModels()
  const ids = ["auto", ...(models.data ?? [])]
  return (
    <section aria-label="Router connection" className="grid gap-3 sm:grid-cols-2 lg:grid-cols-[2fr_1.5fr_1fr_1.5fr]">
      <div className="grid gap-1.5">
        <Label htmlFor="router-url">Router</Label>
        <Input
          id="router-url"
          placeholder={window.location.origin}
          value={settings.routerUrl}
          onChange={(e) => update({ routerUrl: e.target.value })}
          spellCheck={false}
        />
      </div>
      <div className="grid gap-1.5">
        <Label htmlFor="api-key">API key</Label>
        <Input
          id="api-key"
          type="password"
          placeholder="not required"
          value={settings.apiKey}
          onChange={(e) => update({ apiKey: e.target.value })}
          autoComplete="off"
        />
      </div>
      <div className="grid gap-1.5">
        <Label htmlFor="provider">Send to</Label>
        <Select value={settings.provider} onValueChange={(v) => update({ provider: v })}>
          <SelectTrigger id="provider" className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {PROVIDERS.map((p) => (
              <SelectItem key={p} value={p}>
                {p === "auto" ? "Let the router decide" : p}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="grid gap-1.5">
        <Label htmlFor="model">Model</Label>
        <Select value={ids.includes(settings.model) ? settings.model : "auto"} onValueChange={(v) => update({ model: v })}>
          <SelectTrigger id="model" className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {ids.map((id) => (
              <SelectItem key={id} value={id}>
                {id === "auto" ? "Router picks" : id}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </section>
  )
}
