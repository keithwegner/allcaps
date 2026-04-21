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

export type DatasetAdminPayload = HealthPayload & {
  admin: {
    mode: string;
    write_requires_unlock: boolean;
  };
  status: {
    operating_mode: string;
    source_mode: StatusPayload["source_mode"];
    state_root: string;
    db_path: string;
    scheduler_enabled: boolean;
    missing_core_datasets: string[];
    missing_core_dataset_count: number;
    last_refresh: Record<string, unknown>;
    active_job_id: string;
  };
  watchlist_symbols: string[];
  asset_universe: RecordRow[];
  source_manifests: RecordRow[];
  asset_market_manifest: RecordRow[];
  refresh_jobs: RecordRow[];
  job_id?: string;
};

export type DatasetRefreshJobPayload = {
  job_id: string;
  found: boolean;
  job: RecordRow | null;
  recent_jobs: RecordRow[];
};

export type DatasetRefreshRequest = {
  refresh_mode: "bootstrap" | "full" | "incremental";
  remote_url?: string;
  files?: FileList | File[];
};

export type ModelTrainingWorkflow = "single_asset" | "saved_run_portfolio" | "joint_portfolio";

export type ModelTrainingPayload = {
  admin: {
    write_requires_unlock: boolean;
  };
  status: {
    ready: boolean;
    active_job_id: string;
    readiness_errors: Record<string, string[]>;
  };
  defaults: {
    single_asset: Record<string, unknown>;
    saved_run_portfolio: Record<string, unknown>;
    joint_portfolio: Record<string, unknown>;
  };
  asset_options: Array<{ symbol: string; label: string }>;
  feature_versions: string[];
  asset_session_symbols: string[];
  narrative_feature_modes: string[];
  model_families: string[];
  topology_variants: string[];
  run_options: RecordRow[];
  asset_model_runs: RecordRow[];
  recent_jobs: RecordRow[];
  job_id?: string;
};

export type ModelTrainingJobPayload = {
  job_id: string;
  found: boolean;
  job: RecordRow | null;
  recent_jobs: RecordRow[];
};

export type ModelTrainingJobRequest = {
  workflow_mode: ModelTrainingWorkflow;
  run_name?: string;
  target_asset?: string;
  feature_version?: string;
  llm_enabled?: boolean;
  train_window?: number;
  validation_window?: number;
  test_window?: number;
  step_size?: number;
  transaction_cost_bps?: number;
  ridge_alpha?: number;
  threshold_grid?: string;
  minimum_signal_grid?: string;
  account_weight_grid?: string;
  fallback_mode?: "SPY" | "FLAT";
  component_run_ids?: string[];
  selected_symbols?: string[];
  topology_variants?: string[];
  model_families?: string[];
  narrative_feature_modes?: string[];
};

export type RunsPayload = {
  count: number;
  runs: RecordRow[];
};

export type SessionOption = {
  value: string;
  label: string;
};

export type RunDetailPayload = {
  found: boolean;
  run_id: string;
  errors: string[];
  run: RecordRow;
  settings: RecordRow;
  metrics: RecordRow;
  selected_params: Record<string, unknown>;
  model_artifact: Record<string, unknown>;
  charts: {
    equity?: PlotlyFigure;
    benchmarks?: PlotlyFigure;
    diagnostics?: PlotlyFigure;
  };
  tables: {
    benchmarks?: RecordRow[];
    variant_summary?: RecordRow[];
    narrative_lift?: RecordRow[];
    feature_family_summary?: RecordRow[];
    windows?: RecordRow[];
    feature_importance?: RecordRow[];
    diagnostics?: RecordRow[];
    trades?: RecordRow[];
    candidate_predictions?: RecordRow[];
  };
  row_counts: RecordRow;
  session_options: SessionOption[];
  selected_session: {
    session_date?: string;
    prediction?: RecordRow[];
    decision?: RecordRow[];
    candidates?: RecordRow[];
    feature_contributions?: RecordRow[];
    post_attribution?: RecordRow[];
    account_attribution?: RecordRow[];
  };
  leakage_audit: Record<string, unknown>;
};

export type RunComparisonPayload = {
  ready: boolean;
  base_run_id: string;
  run_ids: string[];
  missing_run_ids: string[];
  scorecard: RecordRow[];
  setting_diffs: RecordRow[];
  feature_diffs: RecordRow[];
  benchmark_deltas: RecordRow[];
  change_notes: string[];
  charts: {
    equity?: PlotlyFigure;
  };
};

export type ReplayRunOption = {
  run_id: string;
  run_name: string;
  target_asset: string;
  run_type: string;
  allocator_mode: string;
  created_at?: string | null;
  robust_score?: number;
  total_return?: number;
};

export type ReplaySessionOption = {
  value: string;
  label: string;
  signal_session_date: string;
  post_count: number;
  history_rows_available: number;
};

export type ReplayPayload = {
  ready: boolean;
  message: string;
  selected_run_id: string;
  selected_session_date: string;
  min_history_rows: number;
  run_options: ReplayRunOption[];
  sessions: ReplaySessionOption[];
  summary: RecordRow;
};

export type ReplaySessionPayload = {
  ready: boolean;
  message: string;
  run_id: string;
  run_name?: string;
  target_asset?: string;
  signal_session_date: string;
  metrics: Record<string, unknown>;
  metadata: Record<string, unknown>;
  prediction: RecordRow[];
  comparison_rows: RecordRow[];
  feature_importance: RecordRow[];
  feature_contributions: RecordRow[];
  post_attribution: RecordRow[];
  account_attribution: RecordRow[];
};

export type LivePayload = {
  configured: boolean;
  errors: string[];
  warnings: string[];
  decision: RecordRow | null;
  board: RecordRow[];
};

export type AdminSessionPayload = {
  token: string;
  token_type: string;
  expires_at: string;
  expires_in_seconds: number;
  mode: string;
};

export type LiveConfigSaveRequest = {
  portfolio_run_id: string;
  fallback_mode: "SPY" | "FLAT";
};

export type LiveCapturePayload = {
  persisted_assets: number;
  persisted_decisions: number;
  captured: number;
  settled: number;
  performance_persisted: boolean;
};

export type PaperCurrentActionRequest = {
  action: "enable" | "disable" | "reset" | "archive";
  starting_cash?: number;
};

export type LiveOpsPayload = LivePayload & {
  admin: {
    mode: string;
    write_requires_unlock: boolean;
    capture_scope: string;
  };
  current_config: RecordRow | null;
  seeded_config: RecordRow | null;
  run_options: RecordRow[];
  asset_history: RecordRow[];
  decision_history: RecordRow[];
  paper: {
    current_config: RecordRow | null;
    active_config: RecordRow | null;
    portfolios: RecordRow[];
    decision_journal: RecordRow[];
    trade_ledger: RecordRow[];
    equity_curve: RecordRow[];
    benchmark_curve: RecordRow[];
  };
  capture_result: LiveCapturePayload;
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

export type DiscoveryPayload = {
  ready: boolean;
  message: string;
  source_mode: StatusPayload["source_mode"];
  latest_ranked_at: string | null;
  summary: RecordRow;
  charts: {
    top_discovered_accounts?: PlotlyFigure;
    ranking_history?: PlotlyFigure;
  };
  active_accounts: RecordRow[];
  latest_rankings: RecordRow[];
  override_history: RecordRow[];
  recent_ranking_history: RecordRow[];
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

async function postJson<T>(path: string, payload: unknown = {}, token?: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (body.detail) {
        detail = Array.isArray(body.detail) ? body.detail.join("; ") : String(body.detail);
      }
    } catch {
      // Keep the HTTP status fallback when the response is not JSON.
    }
    throw new Error(`API request failed: ${detail}`);
  }
  return response.json() as Promise<T>;
}

async function postForm<T>(path: string, formData: FormData, token?: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: formData,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (body.detail) {
        detail = Array.isArray(body.detail) ? body.detail.join("; ") : String(body.detail);
      }
    } catch {
      // Keep the HTTP status fallback when the response is not JSON.
    }
    throw new Error(`API request failed: ${detail}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  baseUrl: API_BASE_URL,
  status: () => getJson<StatusPayload>("/api/status"),
  adminSession: (password = "") => postJson<AdminSessionPayload>("/api/admin/session", { password }),
  health: () => getJson<HealthPayload>("/api/datasets/health"),
  datasetAdmin: () => getJson<DatasetAdminPayload>("/api/datasets/admin"),
  saveWatchlist: (symbols: string[], reset: boolean, token: string) =>
    postJson<DatasetAdminPayload>("/api/datasets/watchlist", { symbols, reset }, token),
  startDatasetRefresh: (request: DatasetRefreshRequest, token: string) => {
    const formData = new FormData();
    formData.set("refresh_mode", request.refresh_mode);
    formData.set("remote_url", request.remote_url ?? "");
    Array.from(request.files ?? []).forEach((file) => formData.append("files", file));
    return postForm<DatasetAdminPayload>("/api/datasets/refresh", formData, token);
  },
  datasetRefreshJob: (jobId: string) => getJson<DatasetRefreshJobPayload>(`/api/datasets/jobs/${encodeURIComponent(jobId)}`),
  modelTraining: () => getJson<ModelTrainingPayload>("/api/models/training"),
  startModelTrainingJob: (request: ModelTrainingJobRequest, token: string) =>
    postJson<ModelTrainingPayload>("/api/models/jobs", request, token),
  modelTrainingJob: (jobId: string) => getJson<ModelTrainingJobPayload>(`/api/models/jobs/${encodeURIComponent(jobId)}`),
  runs: () => getJson<RunsPayload>("/api/runs"),
  runDetail: (runId: string, options: { variantName?: string; sessionDate?: string } = {}) => {
    const params = new URLSearchParams();
    if (options.variantName) {
      params.set("variant_name", options.variantName);
    }
    if (options.sessionDate) {
      params.set("session_date", options.sessionDate);
    }
    const query = params.toString();
    return getJson<RunDetailPayload>(`/api/runs/${encodeURIComponent(runId)}${query ? `?${query}` : ""}`);
  },
  runComparison: (runIds: string[], baseRunId?: string) => {
    const params = new URLSearchParams();
    runIds.forEach((runId) => params.append("run_ids", runId));
    if (baseRunId) {
      params.set("base_run_id", baseRunId);
    }
    const query = params.toString();
    return getJson<RunComparisonPayload>(`/api/runs/compare${query ? `?${query}` : ""}`);
  },
  replay: (runId?: string) => {
    const params = new URLSearchParams();
    if (runId) {
      params.set("run_id", runId);
    }
    const query = params.toString();
    return getJson<ReplayPayload>(`/api/replay${query ? `?${query}` : ""}`);
  },
  replaySession: (runId: string, signalSessionDate: string) => {
    const params = new URLSearchParams({ run_id: runId, signal_session_date: signalSessionDate });
    return getJson<ReplaySessionPayload>(`/api/replay/session?${params.toString()}`);
  },
  research: (filters?: ResearchFilters) => getJson<ResearchPayload>(`/api/research${researchQuery(filters)}`),
  researchExportUrl: (filters?: ResearchFilters) => `${API_BASE_URL}/api/research/export${researchQuery(filters)}`,
  discovery: () => getJson<DiscoveryPayload>("/api/discovery"),
  live: () => getJson<LivePayload>("/api/live/current"),
  liveOps: () => getJson<LiveOpsPayload>("/api/live/ops"),
  saveLiveConfig: (request: LiveConfigSaveRequest, token: string) => postJson<LiveOpsPayload>("/api/live/config", request, token),
  captureLive: (token: string) => postJson<LiveOpsPayload>("/api/live/capture", {}, token),
  paperCurrentAction: (request: PaperCurrentActionRequest, token: string) => postJson<LiveOpsPayload>("/api/paper/current", request, token),
  paperPortfolios: () => getJson<PaperPortfoliosPayload>("/api/paper/portfolios"),
  performance: (paperPortfolioId: string) => getJson<PerformancePayload>(`/api/performance/${paperPortfolioId}`),
};
