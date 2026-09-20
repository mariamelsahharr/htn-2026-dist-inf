import { Badge } from "@/components/ui/badge"
import { colorFor } from "@/lib/api"

export function UpstreamBadge({ upstream, className }: { upstream: string; className?: string }) {
  const color = colorFor(upstream)
  return (
    <Badge
      variant="outline"
      className={className}
      style={{ borderColor: color, color, backgroundColor: `${color}1a` }}
    >
      {upstream}
    </Badge>
  )
}
