import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { App } from "../src/App";

vi.mock("plotly.js-dist-min", () => ({ default: {} }));
vi.mock("react-plotly.js/factory", () => ({
  default: () =>
    function MockPlot() {
      return <div data-testid="plotly-chart" />;
    },
}));

const figure = {
  data: [{ x: ["2025-01-03"], y: [1], type: "scatter" }],
  layout: { title: { text: "Chart" } },
};

const statusPayload = {
  title: "AllCaps",
  mode: "private",
  state_root: "/tmp/allcaps",
  db_path: "/tmp/allcaps/workbench.duckdb",
  source_mode: { mode: "truth_only", truth_post_count: 42, x_post_count: 0 },
  missing_core_datasets: [],
  dataset_count: 12,
};

const runsPayload = {
  count: 2,
  runs: [
    {
      run_id: "run-portfolio-1",
      run_name: "Portfolio Alpha",
      run_type: "portfolio_allocator",
      allocator_mode: "joint_model",
      target_asset: "PORTFOLIO",
      total_return: 0.12,
      robust_score: 0.4,
    },
    {
      run_id: "run-asset-1",
      run_name: "SPY Baseline",
      run_type: "asset_model",
      allocator_mode: "",
      target_asset: "SPY",
      robust_score: 0.2,
    },
  ],
};

const runsPayloadExtended = {
  count: 3,
  runs: [
    ...runsPayload.runs,
    {
      run_id: "run-asset-2",
      run_name: "QQQ Baseline",
      run_type: "asset_model",
      allocator_mode: "",
      target_asset: "QQQ",
      robust_score: 0.15,
    },
  ],
};

const healthPayload = {
  summary: {
    overall_severity: "warn",
    severe_count: 0,
    warn_count: 1,
    last_refresh_status: "success",
    last_refresh_mode: "full",
  },
  latest: [{ scope_kind: "dataset", scope_key: "asset_intraday", check_name: "freshness_hours", severity: "warn" }],
  trend: [{ generated_at: "2025-01-03T12:00:00Z", warn_count: 1, severe_count: 0 }],
  refresh_history: [{ refresh_id: "refresh-1", refresh_mode: "full", status: "success" }],
  registry: [{ dataset: "normalized_posts", row_count: 42 }],
};

const datasetAdminPayload = {
  ...healthPayload,
  admin: { mode: "private", write_requires_unlock: true },
  status: {
    operating_mode: "Truth Social-only",
    source_mode: statusPayload.source_mode,
    state_root: statusPayload.state_root,
    db_path: statusPayload.db_path,
    scheduler_enabled: false,
    missing_core_datasets: [],
    missing_core_dataset_count: 0,
    last_refresh: { status: "success" },
    active_job_id: "",
  },
  watchlist_symbols: ["NVDA", "QQQ"],
  asset_universe: [{ symbol: "NVDA", source: "watchlist" }],
  source_manifests: [{ source_name: "truth_archive", status: "success" }],
  asset_market_manifest: [{ symbol: "NVDA", status: "success" }],
  refresh_jobs: [{ job_id: "dataset-refresh-1", refresh_mode: "full", status: "success" }],
};

const datasetAdminAfterRefresh = {
  ...datasetAdminPayload,
  job_id: "dataset-refresh-2",
  refresh_jobs: [
    { job_id: "dataset-refresh-2", refresh_mode: "full", status: "success" },
    ...datasetAdminPayload.refresh_jobs,
  ],
};

const researchPayload = {
  ready: true,
  message: "",
  source_mode: statusPayload.source_mode,
  filters: {
    date_start: "2025-01-01",
    date_end: "2025-01-31",
    platforms: ["Truth Social"],
    include_reshares: false,
    tracked_only: false,
    trump_authored_only: true,
    keyword: "",
    scale_markers: true,
    narrative_platforms: ["Truth Social"],
  },
  headline_metrics: { sessions_with_posts: 2, posts_in_view: 3, truth_posts: 3, mean_sentiment: 0.2 },
  charts: {
    social_activity: figure,
    narrative_frequency: figure,
    narrative_returns: figure,
    narrative_asset_heatmap: figure,
  },
  session_rows: [{ signal_session_date: "2025-01-03", author_handle: "realDonaldTrump", post_count: 2 }],
  post_rows: [{ post_id: "post-1", author_handle: "realDonaldTrump", text: "Markets" }],
  narrative_filter_options: {
    topics: ["All", "markets"],
    policy_buckets: ["All", "trade"],
    stances: ["All", "positive"],
    urgency_bands: ["All", "high"],
    assets: ["All", "SPY"],
    platforms: ["Truth Social"],
    tracked_scopes: ["All posts", "Tracked only"],
    bucket_fields: [{ value: "semantic_topic", label: "Topic" }],
  },
  narrative_metrics: { narrative_tagged_posts: 3, narrative_sessions: 2, cache_hit_rate: 1, providers_used: 1 },
  provider_summary: [{ semantic_provider: "heuristic-fallback", row_count: 3 }],
  narrative_frequency: [{ semantic_topic: "markets", post_count: 3 }],
  narrative_returns: [{ semantic_topic: "markets", next_return: 0.01 }],
  narrative_asset_heatmap: [{ asset_symbol: "SPY", semantic_topic: "markets", post_count: 2 }],
  narrative_posts: [{ post_id: "post-1", semantic_topic: "markets" }],
  narrative_events: [{ signal_session_date: "2025-01-03", semantic_topic: "markets" }],
  export_filename: "research-pack.zip",
};

const researchAssetPayload = {
  ready: true,
  message: "",
  source_mode: statusPayload.source_mode,
  filters: researchPayload.filters,
  controls: {
    selected_asset: "NVDA",
    comparison_mode: "normalized",
    benchmark_symbol: "QQQ",
    pre_sessions: 3,
    post_sessions: 5,
    before_minutes: 120,
    after_minutes: 240,
    intraday_session_date: "2025-01-03",
    intraday_anchor_post_id: "post-1",
  },
  asset_options: [{ symbol: "NVDA", label: "NVDA", source: "watchlist", is_watchlist: true, has_daily: true }],
  benchmark_options: ["None", "QQQ"],
  headline_metrics: { sessions_in_range: 2, asset_move: 0.03, mapped_post_count: 2, intraday_bars: 10 },
  charts: { overlay: figure, event_study: figure, intraday: figure },
  asset_session_rows: [{ signal_session_date: "2025-01-03", asset_symbol: "NVDA" }],
  mapped_post_rows: [{ post_id: "post-1", match_reasons: "semantic_primary_asset" }],
  event_study_rows: [{ relative_session: 0, asset_symbol: "NVDA" }],
  intraday_anchor_options: [{ anchor_id: "post-1", session_date: "2025-01-03", label: "Trump post", post_timestamp: "2025-01-03T14:00:00Z" }],
  intraday_coverage_rows: [{ symbol: "NVDA", bars: 10 }],
  intraday_window_rows: [{ timestamp: "2025-01-03T14:00:00Z", symbol: "NVDA", close: 101 }],
  empty_states: {},
};

const discoveryPayload = {
  ready: true,
  message: "Rankings available",
  source_mode: { mode: "truth_plus_x", truth_post_count: 42, x_post_count: 10 },
  latest_ranked_at: "2025-01-03T12:00:00Z",
  admin: { mode: "private", writes_enabled: true, write_disabled_reason: "" },
  summary: { x_candidate_post_count: 10, active_account_count: 2, latest_ranking_count: 2, override_count: 2 },
  charts: { top_discovered_accounts: figure, ranking_history: figure },
  active_accounts: [{ account_id: "acct-1", handle: "macroalpha" }],
  latest_rankings: [{ account_id: "acct-1", handle: "macroalpha", discovery_score: 0.9, suppressed_by_override: false }],
  override_account_options: [
    { account_id: "acct-1", handle: "macroalpha", display_name: "Macro Alpha", source_platform: "X", status: "candidate", source: "ranking", label: "@macroalpha" },
    { account_id: "acct-muted", handle: "policywatch", display_name: "Policy Watch", source_platform: "X", status: "candidate", source: "ranking", label: "@policywatch" },
  ],
  override_history: [{ override_id: "override-pin", account_id: "acct-1", handle: "macroalpha", action: "pin", effective_from: "2025-01-01" }],
  recent_ranking_history: [{ ranked_at: "2025-01-03T12:00:00Z", handle: "macroalpha", discovery_score: 0.9 }],
};

const runDetailPayload = {
  found: true,
  run_id: "run-portfolio-1",
  errors: [],
  run: { run_id: "run-portfolio-1", run_name: "Portfolio Alpha" },
  settings: {
    run_type: "portfolio_allocator",
    target_asset: "PORTFOLIO",
    deployment_variant: "per_asset_hybrid",
    deployment_narrative_feature_mode: "hybrid",
    allocator_mode: "joint_model",
  },
  metrics: { total_return: 0.12, robust_score: 0.4, max_drawdown: -0.03 },
  selected_params: {},
  model_artifact: { feature_count: 12 },
  charts: { equity: figure, benchmarks: figure, diagnostics: figure },
  tables: {
    variant_summary: [{ variant_name: "per_asset_hybrid", topology: "per_asset", narrative_feature_mode: "hybrid", deployment_winner: true }],
    narrative_lift: [{ narrative_feature_mode: "hybrid", robust_score_lift: 0.1 }],
    feature_family_summary: [{ feature_family: "semantic", contribution_abs_sum: 1.2 }],
    feature_importance: [{ feature: "post_count", importance: 0.5 }],
    diagnostics: [{ metric: "miss", value: 1 }],
    trades: [{ asset_symbol: "QQQ", net_return: 0.01 }],
  },
  row_counts: { trades: 3 },
  session_options: [{ value: "2025-01-03", label: "2025-01-03" }],
  selected_session: {
    decision: [{ winning_asset: "QQQ", stance: "LONG" }],
    prediction: [{ asset_symbol: "QQQ" }],
    candidates: [{ asset_symbol: "QQQ", expected_return_score: 0.03 }],
    feature_contributions: [{ feature: "semantic_score", contribution: 0.1 }],
    post_attribution: [{ author_handle: "realDonaldTrump", text: "Markets" }],
    account_attribution: [{ author_handle: "macroalpha", contribution: 0.2 }],
  },
  leakage_audit: {},
};

const runComparisonPayload = {
  ready: true,
  base_run_id: "run-portfolio-1",
  run_ids: ["run-portfolio-1", "run-asset-1"],
  missing_run_ids: [],
  scorecard: [{ run_name: "SPY Baseline", total_return: 0.04 }],
  setting_diffs: [{ setting: "fallback_mode", run_value: "SPY" }],
  feature_diffs: [{ feature: "semantic_score", status: "added" }],
  benchmark_deltas: [{ benchmark: "SPY", delta: 0.02 }],
  change_notes: ["run-asset-1: robust score changed"],
  charts: { equity: figure },
};

const replayPayload = {
  ready: true,
  message: "",
  selected_run_id: "run-asset-1",
  selected_session_date: "2025-03-03",
  min_history_rows: 20,
  run_options: [
    { run_id: "run-asset-1", run_name: "SPY Baseline", target_asset: "SPY", run_type: "asset_model", allocator_mode: "" },
    { run_id: "run-asset-2", run_name: "QQQ Baseline", target_asset: "QQQ", run_type: "asset_model", allocator_mode: "" },
  ],
  sessions: [
    { value: "2025-03-03", label: "2025-03-03", signal_session_date: "2025-03-03", post_count: 3, history_rows_available: 30 },
    { value: "2025-03-04", label: "2025-03-04", signal_session_date: "2025-03-04", post_count: 4, history_rows_available: 31 },
  ],
  summary: { asset_model_run_count: 1, eligible_session_count: 1 },
};

const replaySessionPayload = {
  ready: true,
  message: "",
  run_id: "run-asset-1",
  run_name: "SPY Baseline",
  target_asset: "SPY",
  signal_session_date: "2025-03-03",
  metrics: {
    target_asset: "SPY",
    replay_session: "2025-03-03",
    replay_score: 0.02,
    replay_confidence: 0.7,
    suggested_stance: "LONG SPY NEXT SESSION",
    training_rows_used: 30,
    full_history_score: 0.021,
    replay_vs_full_history_drift: 0.001,
    actual_next_session_return: 0.005,
  },
  metadata: { full_history_comparison_available: true, deployment_params: { threshold: 0.01, min_post_count: 2 } },
  prediction: [{ signal_session_date: "2025-03-03", expected_return_score: 0.02 }],
  comparison_rows: [{ check: "Replay vs full-history drift", value: 0.001 }],
  feature_importance: [{ feature: "post_count", importance: 0.5 }],
  feature_contributions: [{ feature: "sentiment", contribution: 0.1 }],
  post_attribution: [{ author_handle: "realDonaldTrump", text: "Post" }],
  account_attribution: [{ author_handle: "realDonaldTrump", contribution: 0.3 }],
};

const modelTrainingPayload = {
  admin: { write_requires_unlock: true },
  status: { ready: true, active_job_id: "", readiness_errors: { single_asset: [], saved_run_portfolio: [], joint_portfolio: [] } },
  defaults: { single_asset: {}, saved_run_portfolio: {}, joint_portfolio: { feature_version: "asset-v1", selected_symbols: ["SPY", "QQQ", "NVDA"] } },
  asset_options: [{ symbol: "SPY", label: "SPY" }, { symbol: "QQQ", label: "QQQ" }],
  feature_versions: ["asset-v1"],
  asset_session_symbols: ["SPY", "QQQ", "NVDA"],
  narrative_feature_modes: ["baseline", "hybrid"],
  model_families: ["ridge", "elastic_net"],
  topology_variants: ["per_asset", "pooled"],
  run_options: runsPayloadExtended.runs,
  asset_model_runs: [
    { run_id: "run-asset-1", run_name: "SPY Baseline", target_asset: "SPY", robust_score: 0.2 },
    { run_id: "run-asset-2", run_name: "QQQ Baseline", target_asset: "QQQ", robust_score: 0.15 },
  ],
  recent_jobs: [{ job_id: "model-training-1", workflow_mode: "joint_portfolio", status: "success", run_id: "run-portfolio-1", summary: JSON.stringify({ metrics: { total_return: 0.12 } }) }],
};

const modelTrainingStartedPayload = {
  ...modelTrainingPayload,
  job_id: "model-training-2",
  recent_jobs: [{ job_id: "model-training-2", workflow_mode: "joint_portfolio", status: "success", run_id: "run-portfolio-2", summary: JSON.stringify({ metrics: { total_return: 0.2 } }) }],
};

const liveOpsPayload = {
  configured: true,
  errors: [],
  warnings: [],
  decision: { winning_asset: "QQQ", decision_source: "eligible", winner_score: 0.03, eligible_asset_count: 2 },
  board: [{ asset_symbol: "QQQ", qualifies: true, expected_return_score: 0.03 }, { asset_symbol: "SPY", qualifies: true, expected_return_score: 0.02 }],
  admin: { mode: "private", write_requires_unlock: true, capture_scope: "stored_data_only" },
  current_config: { portfolio_run_id: "run-portfolio-1", fallback_mode: "SPY" },
  seeded_config: null,
  run_options: [{ run_id: "run-portfolio-1", run_name: "Portfolio Alpha", deployment_variant: "per_asset_hybrid", selected_symbols: "SPY,QQQ" }],
  asset_history: [{ generated_at: "2025-01-03T12:00:00Z", asset_symbol: "QQQ", expected_return_score: 0.03 }],
  decision_history: [{ signal_session_date: "2025-01-03", winning_asset: "QQQ" }],
  paper: {
    current_config: { paper_portfolio_id: "paper-1" },
    active_config: { paper_portfolio_id: "paper-1", enabled: true, starting_cash: 100000, transaction_cost_bps: 2 },
    portfolios: [{ paper_portfolio_id: "paper-1", portfolio_run_name: "Portfolio Alpha", enabled: true }],
    decision_journal: [{ paper_portfolio_id: "paper-1", status: "pending", winning_asset: "QQQ" }],
    trade_ledger: [{ asset_symbol: "QQQ", net_return: 0.01 }],
    equity_curve: [{ paper_portfolio_id: "paper-1", equity: 101000 }],
    benchmark_curve: [{ paper_portfolio_id: "paper-1", equity: 100500 }],
  },
  capture_result: { persisted_assets: 2, persisted_decisions: 1, captured: 1, settled: 0, performance_persisted: true },
};

const portfoliosPayload = {
  current_config: { paper_portfolio_id: "paper-1" },
  portfolios: [
    { paper_portfolio_id: "paper-1", portfolio_run_name: "Portfolio Alpha" },
    { paper_portfolio_id: "paper 2", portfolio_run_name: "Portfolio Beta" },
  ],
};

const performancePayload = {
  persisted: true,
  summary: { overall_severity: "ok", total_return: 0.01, alpha: 0.005, fallback_rate: 0.1 },
  diagnostics: [{ metric_name: "score_outcome_correlation", severity: "warn", observed_value: 0.1 }],
  equity_comparison: [],
  rolling_returns: [],
  score_outcomes: [],
  score_buckets: [],
  winner_distribution: [{ winning_asset: "QQQ", decision_count: 7 }],
  drift: [],
};

function payloadFor(pathname: string, method = "GET") {
  if (pathname === "/api/status") return statusPayload;
  if (pathname === "/api/datasets/health") return healthPayload;
  if (pathname === "/api/datasets/admin") return datasetAdminPayload;
  if (pathname === "/api/datasets/watchlist") return datasetAdminPayload;
  if (pathname === "/api/datasets/refresh") return datasetAdminAfterRefresh;
  if (pathname === "/api/datasets/jobs/dataset-refresh-2") return { job_id: "dataset-refresh-2", found: true, job: datasetAdminAfterRefresh.refresh_jobs[0], recent_jobs: datasetAdminAfterRefresh.refresh_jobs };
  if (pathname === "/api/runs") return runsPayloadExtended;
  if (pathname === "/api/models/training") return modelTrainingPayload;
  if (pathname === "/api/models/jobs") return modelTrainingStartedPayload;
  if (pathname === "/api/models/jobs/model-training-2") return { job_id: "model-training-2", found: true, job: modelTrainingStartedPayload.recent_jobs[0], recent_jobs: modelTrainingStartedPayload.recent_jobs };
  if (pathname === "/api/runs/run-portfolio-1") return runDetailPayload;
  if (pathname === "/api/runs/run-asset-1") return { ...runDetailPayload, run_id: "run-asset-1", run: { run_id: "run-asset-1", run_name: "SPY Baseline" }, settings: { run_type: "asset_model", target_asset: "SPY" } };
  if (pathname === "/api/runs/run-asset-2") return { ...runDetailPayload, run_id: "run-asset-2", run: { run_id: "run-asset-2", run_name: "QQQ Baseline" }, settings: { run_type: "asset_model", target_asset: "QQQ" } };
  if (pathname === "/api/runs/compare") return runComparisonPayload;
  if (pathname === "/api/replay") return replayPayload;
  if (pathname === "/api/replay/session") return replaySessionPayload;
  if (pathname === "/api/research/assets") return researchAssetPayload;
  if (pathname === "/api/research") return researchPayload;
  if (pathname === "/api/discovery") return discoveryPayload;
  if (pathname === "/api/admin/session") return { token: "admin-token", token_type: "bearer", expires_at: "2026-04-21T12:00:00Z", expires_in_seconds: 43200, mode: "private" };
  if (pathname === "/api/discovery/overrides" && method === "POST") return discoveryPayload;
  if (pathname.startsWith("/api/discovery/overrides/") && method === "DELETE") return { ...discoveryPayload, override_history: [] };
  if (pathname === "/api/live/current") return liveOpsPayload;
  if (pathname === "/api/live/ops") return liveOpsPayload;
  if (pathname === "/api/live/config") return liveOpsPayload;
  if (pathname === "/api/live/capture") return liveOpsPayload;
  if (pathname === "/api/paper/current") return liveOpsPayload;
  if (pathname === "/api/paper/portfolios") return portfoliosPayload;
  if (pathname === "/api/performance/paper-1") return performancePayload;
  if (pathname === "/api/performance/paper%202") return { ...performancePayload, summary: { ...performancePayload.summary, total_return: 0.02 } };
  return { error: `No fixture for ${method} ${pathname}` };
}

function installFetchMock() {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), "http://127.0.0.1:5173");
    const method = init?.method ?? "GET";
    const payload = payloadFor(url.pathname, method);
    return new Response(JSON.stringify(payload), {
      status: "error" in payload ? 404 : 200,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function renderApp() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: Infinity,
        refetchOnWindowFocus: false,
      },
      mutations: {
        retry: false,
      },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>,
  );
}

describe("App component", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    installFetchMock();
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  test("renders every primary page from mocked API data", async () => {
    const user = userEvent.setup();
    renderApp();

    expect(await screen.findByRole("heading", { name: "Web-first decision workbench" })).toBeInTheDocument();
    expect(await screen.findByText("truth_only (42 Truth, 0 X)")).toBeInTheDocument();
    expect(screen.getByText("Portfolio Alpha")).toBeInTheDocument();
    expect(screen.getByText("Workflow map")).toBeInTheDocument();
    expect(screen.getByText("Explore")).toBeInTheDocument();
    expect(screen.getByText("Build")).toBeInTheDocument();
    expect(screen.getByText("Operate")).toBeInTheDocument();
    expect(screen.getAllByText(/Start here to confirm the API/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Follow the flow from data and research/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Research: Sentiment/ }));
    expect(await screen.findByRole("heading", { name: "Research workspace" })).toBeInTheDocument();
    expect(screen.getByText(/Truth Social-only mode is active/)).toBeInTheDocument();
    expect(screen.getAllByText(/Use this page to inspect filtered Trump Truth Social/).length).toBeGreaterThan(0);
    expect(screen.getByRole("link", { name: "Export research pack" })).toHaveAttribute("href", expect.stringContaining("/api/research/export"));
    expect(await screen.findByRole("heading", { name: "Multi-asset comparison, event study, and intraday reaction" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Discovery: Tracked/ }));
    expect(await screen.findByRole("heading", { name: "Discovery workspace" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Discovery admin overrides" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Run Explorer: Saved/ }));
    expect(await screen.findByRole("heading", { name: "Run Explorer" })).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Variant comparison" })).toBeInTheDocument();
    expect(screen.getByText("run-asset-1: robust score changed")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Replay: Historical/ }));
    expect(await screen.findByRole("heading", { name: "Historical Replay Workspace" })).toBeInTheDocument();
    expect(await screen.findByText("LONG SPY NEXT SESSION")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Model Training: Train/ }));
    expect(await screen.findByRole("heading", { name: "Model Training Job Console" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Use Joint Portfolio workflow" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Data Admin: Refresh/ }));
    expect(await screen.findByRole("heading", { name: "Data Admin Console" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Watchlist controls" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Live Ops: Operate/ }));
    expect(await screen.findByRole("heading", { name: "Live Ops Console" })).toBeInTheDocument();
    expect(screen.getByText("Stored-data capture only")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Paper \+ Performance: Portfolio/ }));
    expect(await screen.findByRole("heading", { name: "Portfolio selector" })).toBeInTheDocument();
    expect(await screen.findByText("score_outcome_correlation")).toBeInTheDocument();
  });

  test("supports admin mutations from React pages", async () => {
    const user = userEvent.setup();
    const fetchMock = installFetchMock();
    renderApp();

    await user.click(await screen.findByRole("button", { name: /Data Admin: Refresh/ }));
    await user.type(screen.getByPlaceholderText(/password/i), "secret");
    await user.click(screen.getByRole("button", { name: "Unlock admin writes" }));
    expect(await screen.findByText("Unlocked for this browser session")).toBeInTheDocument();
    await user.clear(screen.getByLabelText("Watchlist symbols"));
    await user.type(screen.getByLabelText("Watchlist symbols"), "NVDA, TSLA");
    await user.click(screen.getByRole("button", { name: "Save watchlist" }));
    await user.type(screen.getByLabelText("Remote X / mention CSV URL"), "https://example.invalid/mentions.csv");
    await user.click(screen.getByRole("button", { name: "Start refresh job" }));
    expect(await screen.findByRole("cell", { name: "dataset-refresh-2" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Discovery: Tracked/ }));
    await user.selectOptions(screen.getByLabelText("Override account"), "acct-muted");
    await user.selectOptions(screen.getByLabelText("Override action"), "suppress");
    await user.type(screen.getByPlaceholderText("Optional rationale"), "Noisy source");
    await user.click(screen.getByRole("button", { name: "Save discovery override" }));
    await user.selectOptions(screen.getByLabelText("Override to delete"), "override-pin");
    await user.click(screen.getByRole("button", { name: "Delete selected override" }));
    await waitFor(() => expect(screen.queryByText("override-pin")).not.toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: /Model Training: Train/ }));
    await user.click(screen.getByRole("button", { name: "Use Single Asset workflow" }));
    expect(screen.getByRole("heading", { name: "Single Asset configuration" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Use Saved-Run Portfolio workflow" }));
    expect(screen.getByRole("heading", { name: "Saved-Run Portfolio configuration" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Use Joint Portfolio workflow" }));
    await user.click(screen.getByRole("button", { name: "Start model training job" }));
    expect(await screen.findByRole("heading", { name: "Latest training result" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Live Ops: Operate/ }));
    await user.click(screen.getByRole("button", { name: "Save pinned portfolio run" }));
    await user.click(screen.getByRole("button", { name: "Capture current board" }));
    await user.click(screen.getByRole("button", { name: "Disable paper trading" }));

    const requestedPaths = fetchMock.mock.calls.map(([input]) => new URL(String(input), "http://127.0.0.1:5173").pathname);
    expect(requestedPaths).toContain("/api/datasets/watchlist");
    expect(requestedPaths).toContain("/api/datasets/refresh");
    expect(requestedPaths).toContain("/api/discovery/overrides");
    expect(requestedPaths).toContain("/api/models/jobs");
    expect(requestedPaths).toContain("/api/live/config");
    expect(requestedPaths).toContain("/api/live/capture");
    expect(requestedPaths).toContain("/api/paper/current");
  });

  test("drives read-only research, run, replay, and performance controls", async () => {
    const user = userEvent.setup();
    renderApp();

    await user.click(await screen.findByRole("button", { name: /Research: Sentiment/ }));
    await user.clear(screen.getByLabelText("Start date"));
    await user.type(screen.getByLabelText("Start date"), "2025-01-02");
    await user.clear(screen.getByLabelText("End date"));
    await user.type(screen.getByLabelText("End date"), "2025-01-30");
    await user.selectOptions(screen.getByLabelText("Platforms"), ["Truth Social", "X"]);
    await user.type(screen.getByPlaceholderText("Optional exact text filter"), "tariff");
    await user.click(screen.getByLabelText("Trump-authored only"));
    await user.click(screen.getByLabelText("Include reshares"));
    await user.click(screen.getByLabelText("Trump + tracked only"));
    await user.click(screen.getByLabelText("Scale activity"));
    await user.selectOptions(screen.getByLabelText("Topic"), "markets");
    await user.selectOptions(screen.getByLabelText("Policy bucket"), "trade");
    await user.selectOptions(screen.getByLabelText("Stance"), "positive");
    await user.selectOptions(screen.getByLabelText("Urgency"), "high");
    await user.selectOptions(screen.getByLabelText("Asset target"), "SPY");
    await user.selectOptions(screen.getByLabelText("Return bucket"), "semantic_topic");
    await user.selectOptions(screen.getByLabelText("Narrative platforms"), ["Truth Social"]);
    await user.selectOptions(screen.getByLabelText("Tracked scope"), "Tracked only");
    await user.selectOptions(screen.getByLabelText("Selected asset"), "NVDA");
    await user.selectOptions(screen.getByLabelText("Comparison mode"), "price");
    await user.selectOptions(screen.getByLabelText("ETF baseline"), "QQQ");
    await user.selectOptions(screen.getByLabelText("Intraday anchor"), "post-1");
    await user.clear(screen.getByLabelText("Sessions before"));
    await user.type(screen.getByLabelText("Sessions before"), "2");
    await user.clear(screen.getByLabelText("Sessions after"));
    await user.type(screen.getByLabelText("Sessions after"), "4");
    await user.clear(screen.getByLabelText("Minutes before"));
    await user.type(screen.getByLabelText("Minutes before"), "90");
    await user.clear(screen.getByLabelText("Minutes after"));
    await user.type(screen.getByLabelText("Minutes after"), "300");
    expect(await screen.findByRole("heading", { name: "Intraday reaction" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Run Explorer: Saved/ }));
    await user.selectOptions(await screen.findByLabelText("Run type"), "asset_model");
    await user.selectOptions(screen.getByLabelText("Target asset"), "QQQ");
    await user.type(screen.getByPlaceholderText("Run name, id, or asset"), "Baseline");
    await user.selectOptions(screen.getByLabelText("Selected run"), "run-asset-2");
    await user.selectOptions(screen.getByLabelText("Variant"), "per_asset_hybrid");
    await user.selectOptions(screen.getByLabelText("Session inspector"), "2025-01-03");
    await user.selectOptions(screen.getByLabelText("Compare runs"), ["run-portfolio-1", "run-asset-2"]);
    await user.selectOptions(screen.getByLabelText("Base run"), "run-asset-2");
    await waitFor(() => expect(screen.getAllByText("QQQ Baseline").length).toBeGreaterThan(0));

    await user.click(screen.getByRole("button", { name: /Replay: Historical/ }));
    await user.selectOptions(await screen.findByLabelText("Replay template run"), "run-asset-2");
    await user.selectOptions(screen.getByLabelText("Historical signal session"), "2025-03-04");
    expect(await screen.findByText("LONG SPY NEXT SESSION")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Paper \+ Performance: Portfolio/ }));
    await user.selectOptions(await screen.findByRole("combobox"), "paper 2");
    expect(await screen.findByText("score_outcome_correlation")).toBeInTheDocument();
  });

  test("drives admin configuration controls across model, data, and live ops", async () => {
    const user = userEvent.setup();
    renderApp();

    await user.click(await screen.findByRole("button", { name: /Model Training: Train/ }));
    await user.type(screen.getByPlaceholderText("Admin password"), "secret");
    await user.click(screen.getByRole("button", { name: "Unlock admin writes" }));
    expect(await screen.findByText("Unlocked for this browser session")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Use Single Asset workflow" }));
    await user.clear(screen.getByLabelText("Run name"));
    await user.type(screen.getByLabelText("Run name"), "single-asset-smoke");
    await user.selectOptions(screen.getByLabelText("Target asset"), "QQQ");
    await user.selectOptions(screen.getByLabelText("Feature version"), "asset-v1");
    await user.click(screen.getByLabelText("Use semantic rows"));
    await user.clear(screen.getByLabelText("Train window"));
    await user.type(screen.getByLabelText("Train window"), "100");
    await user.clear(screen.getByLabelText("Validation window"));
    await user.type(screen.getByLabelText("Validation window"), "35");
    await user.clear(screen.getByLabelText("Test window"));
    await user.type(screen.getByLabelText("Test window"), "40");
    await user.clear(screen.getByLabelText("Step size"));
    await user.type(screen.getByLabelText("Step size"), "20");
    await user.clear(screen.getByLabelText("Transaction cost bps"));
    await user.type(screen.getByLabelText("Transaction cost bps"), "3");
    await user.clear(screen.getByLabelText("Ridge alpha"));
    await user.type(screen.getByLabelText("Ridge alpha"), "2");
    await user.clear(screen.getByLabelText("Threshold grid"));
    await user.type(screen.getByLabelText("Threshold grid"), "0,0.01");
    await user.clear(screen.getByLabelText("Min post grid"));
    await user.type(screen.getByLabelText("Min post grid"), "1,3");
    await user.clear(screen.getByLabelText("Account weight grid"));
    await user.type(screen.getByLabelText("Account weight grid"), "1,2");

    await user.click(screen.getByRole("button", { name: "Use Saved-Run Portfolio workflow" }));
    await user.selectOptions(screen.getByLabelText("Component runs"), ["run-asset-1", "run-asset-2"]);
    await user.selectOptions(screen.getByLabelText("Fallback mode"), "FLAT");
    await user.click(screen.getByRole("button", { name: "Use Joint Portfolio workflow" }));
    await user.clear(screen.getByLabelText("Selected symbols"));
    await user.type(screen.getByLabelText("Selected symbols"), "SPY, QQQ");
    await user.selectOptions(screen.getByLabelText("Topology variants"), ["pooled"]);
    await user.selectOptions(screen.getByLabelText("Model families"), ["ridge"]);
    await user.selectOptions(screen.getByLabelText("Narrative modes"), ["hybrid"]);
    await user.click(screen.getByRole("button", { name: "Start model training job" }));
    expect(await screen.findByText("run-portfolio-2")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Data Admin: Refresh/ }));
    await user.click(screen.getByRole("button", { name: "Reset watchlist" }));
    await user.selectOptions(screen.getByLabelText("Refresh mode"), "full");
    const file = new File(["author,text\nmacro,hello\n"], "mentions.csv", { type: "text/csv" });
    await user.upload(screen.getByLabelText("CSV upload"), file);

    await user.click(screen.getByRole("button", { name: /Live Ops: Operate/ }));
    await user.selectOptions(screen.getByLabelText("Pinned joint portfolio run"), "run-portfolio-1");
    await user.selectOptions(screen.getByLabelText("Fallback mode"), "FLAT");
    await user.clear(screen.getByLabelText("Starting cash"));
    await user.type(screen.getByLabelText("Starting cash"), "125000");
    await user.click(screen.getByRole("button", { name: "Enable paper trading" }));
    await user.click(screen.getByRole("button", { name: "Archive current portfolio" }));
    await user.clear(screen.getByLabelText("Reset cash"));
    await user.type(screen.getByLabelText("Reset cash"), "90000");
    await user.click(screen.getByRole("button", { name: "Reset portfolio" }));
    await waitFor(() => expect(screen.getAllByText("paper-1").length).toBeGreaterThan(0));
  });

  test("shows empty and error states without crashing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const path = new URL(String(input), "http://127.0.0.1:5173").pathname;
        if (path === "/api/status") {
          return new Response(JSON.stringify(statusPayload), { status: 200, headers: { "Content-Type": "application/json" } });
        }
        if (path === "/api/runs") {
          return new Response(JSON.stringify({ count: 0, runs: [] }), { status: 200, headers: { "Content-Type": "application/json" } });
        }
        return new Response(JSON.stringify({ detail: "broken" }), { status: 500, statusText: "Server Error", headers: { "Content-Type": "application/json" } });
      }),
    );
    const user = userEvent.setup();
    renderApp();

    expect(await screen.findByText("No rows returned yet.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Run Explorer: Saved/ }));
    expect(await screen.findByText("No saved runs have been created yet. Train a model in the Model Training tab first, then return here.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Research: Sentiment/ }));
    expect(await screen.findByText("API request failed: broken")).toBeInTheDocument();
  });

  test("renders a table empty state and limited columns", async () => {
    renderApp();
    const recentRunsHeading = await screen.findByRole("heading", { name: "Recent saved runs" });
    const recentRuns = recentRunsHeading.closest("article");
    expect(recentRuns).not.toBeNull();
    expect(within(recentRuns as HTMLElement).getByRole("columnheader", { name: "run id" })).toBeInTheDocument();
  });
});
