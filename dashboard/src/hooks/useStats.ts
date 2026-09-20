import { useQuery } from "@tanstack/react-query"
import { getModels, getStats } from "@/lib/api"

export const STATS_INTERVAL_MS = 2000

export function useStats() {
  return useQuery({
    queryKey: ["stats"],
    queryFn: getStats,
    refetchInterval: STATS_INTERVAL_MS,
    refetchIntervalInBackground: true,
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
