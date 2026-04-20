const API_BASE_URL = import.meta.env.VITE_ALLCAPS_API_BASE_URL ?? "http://127.0.0.1:8000";

export type RecordRow = Record<string, string | number | boolean | null>;
export type PlotlyFigure = {
  data?: unknown[];
  layout?: Record<string, unknown>;
  frames?: unknown[];
};

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

export type ResearchFilters = {
  date_start?: string;
  date_end?: string;
  platforms?: string[];
  include_reshares?: boolean;
  tracked_only?: boolean;
  trump_authored_only?: boolean;
  keyword?: string;
  scale_markers?: boolean;
  narrative_topic?: string;
  narrative_policy?: string;
  narrative_stance?: string;
  narrative_urgency?: string;
  narrative_asset?: string;
  narrative_platforms?: string[];
  narrative_tracked_scope?: string;
  narrative_bucket_field?: string;
};

export type ResearchPayload = {
  ready: boolean;
  message: string;
  source_mode: StatusPayload["source_mode"];
  filters: ResearchFilters & {
    date_start: string;
    date_end: string;
    platforms: string[];
    include_reshares: boolean;
    tracked_only: boolean;
    trump_authored_only: boolean;
    keyword: string;
    scale_markers: boolean;
    narrative_platforms: string[];
  };
  headline_metrics: RecordRow;
  charts: {
    social_activity?: PlotlyFigure;
    narrative_frequency?: PlotlyFigure;
    narrative_returns?: PlotlyFigure;
    narrative_asset_heatmap?: PlotlyFigure;
  };
  session_rows: RecordRow[];
  post_rows: RecordRow[];
  narrative_filter_options: {
    topics?: string[];
    policy_buckets?: string[];
    stances?: string[];
    urgency_bands?: string[];
    assets?: string[];
    platforms?: string[];
    tracked_scopes?: string[];
    bucket_fields?: Array<{ value: string; label: string }>;
  };
  narrative_metrics: RecordRow;
  provider_summary: RecordRow[];
  narrative_frequency: RecordRow[];
  narrative_returns: RecordRow[];
  narrative_asset_heatmap: RecordRow[];
  narrative_posts: RecordRow[];
  narrative_events: RecordRow[];
  export_filename: string;
};

function researchQuery(filters: ResearchFilters = {}): string {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") {
      return;
    }
    if (Array.isArray(value)) {
      value.forEach((item) => {
        if (item !== "") {
          params.append(key, item);
        }
      });
      return;
    }
    params.set(key, String(value));
  });
  const query = params.toString();
  return query ? `?${query}` : "";
}

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
  research: (filters?: ResearchFilters) => getJson<ResearchPayload>(`/api/research${researchQuery(filters)}`),
  researchExportUrl: (filters?: ResearchFilters) => `${API_BASE_URL}/api/research/export${researchQuery(filters)}`,
  live: () => getJson<LivePayload>("/api/live/current"),
  paperPortfolios: () => getJson<PaperPortfoliosPayload>("/api/paper/portfolios"),
  performance: (paperPortfolioId: string) => getJson<PerformancePayload>(`/api/performance/${paperPortfolioId}`),
};
