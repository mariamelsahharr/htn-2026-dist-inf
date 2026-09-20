import { useQuery } from "@tanstack/react-query"
import { getModels, getStats } from "@/lib/api"
import { useSettings } from "@/lib/settings"

export const STATS_INTERVAL_MS = 2000

export function useStats() {
  const { settings } = useSettings()
  return useQuery({
    queryKey: ["stats", settings.routerUrl, settings.apiKey],
    queryFn: () => getStats(settings),
    refetchInterval: STATS_INTERVAL_MS,
    refetchIntervalInBackground: true,
    staleTime: STATS_INTERVAL_MS,
    retry: false,
  })
}

export function useModels() {
  const { settings } = useSettings()
  return useQuery({
    queryKey: ["models", settings.routerUrl, settings.apiKey],
    queryFn: () => getModels(settings),
    staleTime: 60_000,
    retry: false,
  })
}
