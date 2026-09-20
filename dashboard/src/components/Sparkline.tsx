// A small inline line of recent values; no axes, the endpoint emphasised. Pure SVG.
export function Sparkline({ values, color, width = 120, height = 28 }: { values: number[]; color: string; width?: number; height?: number }) {
  if (values.length < 2) return null
  const max = Math.max(...values, 1)
  const step = width / (values.length - 1)
  const y = (v: number) => height - 2 - (v / max) * (height - 4)
  const points = values.map((v, i) => `${(i * step).toFixed(1)},${y(v).toFixed(1)}`).join(" ")
  const last = values[values.length - 1]
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true" className="shrink-0">
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={(values.length - 1) * step} cy={y(last)} r="2.5" fill={color} />
    </svg>
  )
}
