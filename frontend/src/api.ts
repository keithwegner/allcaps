const API_BASE_URL = import.meta.env.VITE_ALLCAPS_API_BASE_URL ?? "http://127.0.0.1:8000";

export type RecordRow = Record<string, string | number | boolean | null>;

export type StatusPayload = {
  title: string;
  mode: string;
  state_root: string;
  db_path: string;
  source_mode: {
    mode: string;
    truth_post_count: number;
    x_post_count: number;
  };
  missing_core_datasets: string[];
  dataset_count: number;
};

export type HealthPayload = {
  summary: RecordRow;
  latest: RecordRow[];
  trend: RecordRow[];
  refresh_history: RecordRow[];
  registry: RecordRow[];
};

export type RunsPayload = {
  count: number;
  runs: RecordRow[];
};

export type LivePayload = {
  configured: boolean;
  errors: string[];
  warnings: string[];
  decision: RecordRow | null;
  board: RecordRow[];
};

export type PaperPortfoliosPayload = {
  current_config: RecordRow | null;
  portfolios: RecordRow[];
};

export type PerformancePayload = {
  persisted: boolean;
  summary: RecordRow;
  diagnostics: RecordRow[];
  equity_comparison: RecordRow[];
  rolling_returns: RecordRow[];
  score_outcomes: RecordRow[];
  score_buckets: RecordRow[];
  winner_distribution: RecordRow[];
  drift: RecordRow[];
};

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`API request failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  baseUrl: API_BASE_URL,
  status: () => getJson<StatusPayload>("/api/status"),
  health: () => getJson<HealthPayload>("/api/datasets/health"),
  runs: () => getJson<RunsPayload>("/api/runs"),
  live: () => getJson<LivePayload>("/api/live/current"),
  paperPortfolios: () => getJson<PaperPortfoliosPayload>("/api/paper/portfolios"),
  performance: (paperPortfolioId: string) => getJson<PerformancePayload>(`/api/performance/${paperPortfolioId}`),
};
