import { useQuery } from "@tanstack/react-query"
import { getModels, getStats } from "@/lib/api"

export const STATS_INTERVAL_MS = 2000
export const STATS_KEY = ["stats"] as const

export function useStats() {
  return useQuery({
    queryKey: STATS_KEY,
    queryFn: getStats,
    refetchInterval: STATS_INTERVAL_MS,
    staleTime: STATS_INTERVAL_MS,
    retry: false,
  })
}

export function useModels() {
  return useQuery({
    queryKey: ["models"],
    queryFn: getModels,
    staleTime: 60_000,
    retry: false,
  })
}
