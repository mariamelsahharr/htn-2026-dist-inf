import { Badge } from "@/components/ui/badge"
import { tierLabel, tierVars } from "@/lib/api"

export function UpstreamBadge({ upstream, className = "" }: { upstream: string; className?: string }) {
  return (
    <Badge variant="outline" className={`tier-text tier-border tier-bg ${className}`} style={tierVars(upstream)}>
      {tierLabel(upstream)}
    </Badge>
  )
}
