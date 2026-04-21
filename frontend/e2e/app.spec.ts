import { expect, test, type Page } from "@playwright/test";

const statusPayload = {
  title: "Trump Social Trading Research Workbench",
  mode: "private",
  state_root: "/tmp/allcaps-test-state",
  db_path: "/tmp/allcaps-test-state/.workbench/workbench.duckdb",
  source_mode: {
    mode: "truth_only",
    truth_post_count: 42,
    x_post_count: 0,
  },
  missing_core_datasets: [],
  dataset_count: 9,
};

const healthPayload = {
  summary: {
    overall_severity: "warn",
    severe_count: 0,
    warn_count: 1,
    last_refresh_status: "success",
  },
  latest: [
    {
      scope_kind: "dataset",
      scope_key: "asset_intraday",
      check_name: "freshness_hours",
      severity: "warn",
      observed_value: 7.5,
      detail: "Latest intraday row is stale.",
    },
    {
      scope_kind: "dataset",
      scope_key: "normalized_posts",
      check_name: "dataset_presence",
      severity: "ok",
      observed_value: 42,
      detail: "",
    },
  ],
  trend: [],
  refresh_history: [
    {
      refresh_mode: "incremental",
      status: "success",
    },
  ],
  registry: [
    {
      dataset_name: "normalized_posts",
      row_count: 42,
      updated_at: "2026-04-20T12:00:00Z",
    },
  ],
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
    },
    {
      run_id: "run-asset-1",
      run_name: "SPY Baseline",
      run_type: "asset_model",
      allocator_mode: "",
      target_asset: "SPY",
    },
  ],
};

const runDetailPayload = {
  found: true,
  run_id: "run-portfolio-1",
  errors: [],
  run: {
    run_id: "run-portfolio-1",
    run_name: "Portfolio Alpha",
    target_asset: "PORTFOLIO",
  },
  settings: {
    run_type: "portfolio_allocator",
    allocator_mode: "joint_model",
    target_asset: "PORTFOLIO",
    deployment_variant: "per_asset_hybrid",
    deployment_narrative_feature_mode: "hybrid",
  },
  metrics: {
    total_return: 0.12,
    robust_score: 2.1,
    max_drawdown: -0.025,
    trade_count: 8,
  },
  selected_params: {
    deployment_variant: "per_asset_hybrid",
  },
  model_artifact: {
    feature_count: 12,
    model_version: "portfolio-bundle",
  },
  charts: {
    equity: {
      data: [{ type: "scatter", mode: "lines", x: ["2026-04-17"], y: [1.12], name: "Strategy equity" }],
      layout: { title: { text: "Walk-forward out-of-sample equity curve" } },
    },
    benchmarks: {
      data: [{ type: "scatter", mode: "lines", x: ["2026-04-17"], y: [1.08], name: "always_long_spy" }],
      layout: { title: { text: "Strategy vs. benchmark equity curves" } },
    },
    diagnostics: {
      data: [{ type: "scatter", mode: "lines+markers", x: ["2026-04-17"], y: [0.024], name: "Winner score" }],
      layout: { title: { text: "Portfolio allocator diagnostics" } },
    },
  },
  tables: {
    benchmarks: [
      { benchmark_name: "strategy", total_return: 0.12 },
      { benchmark_name: "always_long_spy", total_return: 0.08 },
    ],
    variant_summary: [
      {
        variant_name: "per_asset_baseline",
        topology: "per_asset",
        narrative_feature_mode: "baseline",
        validation_robust_score: 1.1,
        deployment_winner: false,
      },
      {
        variant_name: "per_asset_hybrid",
        topology: "per_asset",
        narrative_feature_mode: "hybrid",
        validation_robust_score: 1.6,
        deployment_winner: true,
      },
    ],
    narrative_lift: [
      {
        variant_name: "per_asset_hybrid",
        topology: "per_asset",
        narrative_feature_mode: "hybrid",
        validation_robust_lift: 0.5,
      },
    ],
    feature_family_summary: [
      {
        feature_family: "semantic",
        feature_count: 4,
        total_importance: 0.72,
      },
      {
        feature_family: "policy",
        feature_count: 2,
        total_importance: 0.22,
      },
    ],
    windows: [{ variant_name: "per_asset_hybrid", window_id: 1, model_family: "ridge" }],
    feature_importance: [{ feature_name: "semantic_relevance_avg", importance: 0.72 }],
    diagnostics: [{ signal_session_date: "2026-04-17", winner_score: 0.024 }],
    trades: [{ next_session_date: "2026-04-20", selected_asset: "QQQ", net_return: 0.014 }],
    candidate_predictions: [{ asset_symbol: "QQQ", expected_return_score: 0.024 }],
  },
  row_counts: {
    trades: 8,
    predictions: 8,
    candidate_predictions: 16,
  },
  session_options: [{ value: "2026-04-17", label: "2026-04-17 | winner QQQ | score +2.400% | runner-up SPY" }],
  selected_session: {
    session_date: "2026-04-17",
    decision: [{ signal_session_date: "2026-04-17", winning_asset: "QQQ", winner_score: 0.024 }],
    candidates: [
      { asset_symbol: "QQQ", expected_return_score: 0.024, is_winner: true },
      { asset_symbol: "SPY", expected_return_score: 0.012, is_winner: false },
    ],
    feature_contributions: [],
    post_attribution: [],
    account_attribution: [],
  },
  leakage_audit: {
    overall_pass: true,
  },
};

const runComparisonPayload = {
  ready: true,
  base_run_id: "run-portfolio-1",
  run_ids: ["run-portfolio-1", "run-asset-1"],
  missing_run_ids: [],
  scorecard: [
    { run_id: "run-portfolio-1", run_name: "Portfolio Alpha", robust_score: 2.1, delta_robust_score_vs_base: 0 },
    { run_id: "run-asset-1", run_name: "SPY Baseline", robust_score: 1.2, delta_robust_score_vs_base: -0.9 },
  ],
  setting_diffs: [{ setting: "run_type", "run-portfolio-1": "portfolio_allocator", "run-asset-1": "asset_model" }],
  feature_diffs: [{ run_id: "run-portfolio-1", feature_count: 12, semantic_features: 4 }],
  benchmark_deltas: [{ run_id: "run-asset-1", benchmark_name: "always_long_spy", delta_total_return_vs_base: -0.02 }],
  change_notes: ["run-asset-1: robust score -0.900; total return -4.00%; run type portfolio_allocator -> asset_model."],
  charts: {
    equity: {
      data: [{ type: "scatter", mode: "lines", x: ["2026-04-17"], y: [1.12], name: "run-portfolio-1" }],
      layout: { title: { text: "Selected run equity curves" } },
    },
  },
};

const researchPayload = {
  ready: true,
  message: "",
  source_mode: {
    mode: "truth_only",
    truth_post_count: 42,
    x_post_count: 0,
  },
  filters: {
    date_start: "2025-01-20",
    date_end: "2026-04-20",
    platforms: ["Truth Social"],
    include_reshares: false,
    tracked_only: false,
    trump_authored_only: true,
    keyword: "",
    scale_markers: true,
    narrative_topic: "All",
    narrative_policy: "All",
    narrative_stance: "All",
    narrative_urgency: "All",
    narrative_asset: "All",
    narrative_platforms: ["Truth Social"],
    narrative_tracked_scope: "All posts",
    narrative_bucket_field: "semantic_topic",
  },
  headline_metrics: {
    sessions_with_posts: 12,
    posts_in_view: 42,
    truth_posts: 42,
    tracked_x_posts: 0,
    mean_sentiment: 0.18,
    sp500_change: 0.04,
  },
  charts: {
    social_activity: {
      data: [{ type: "scatter", mode: "lines", x: ["2026-04-17"], y: [6800], name: "S&P 500 close" }],
      layout: { title: { text: "Research View: social activity vs. market baseline" } },
    },
    narrative_frequency: {
      data: [{ type: "bar", x: ["2026-04-17"], y: [3], name: "markets" }],
      layout: { title: { text: "Narrative frequency over time" } },
    },
    narrative_returns: {
      data: [{ type: "bar", x: ["markets"], y: [0.01], name: "returns" }],
      layout: { title: { text: "Next-session return by narrative bucket" } },
    },
    narrative_asset_heatmap: {
      data: [{ type: "heatmap", x: ["SPY"], y: ["markets"], z: [[3]] }],
      layout: { title: { text: "Asset-by-narrative heatmap" } },
    },
  },
  session_rows: [
    {
      trade_date: "2026-04-17",
      post_count: 3,
      truth_posts: 3,
      sentiment_avg: 0.18,
      sp500_close: 6800,
    },
  ],
  post_rows: [
    {
      source_platform: "Truth Social",
      author_handle: "realDonaldTrump",
      post_time_et: "2026-04-17 08:00",
      sentiment_score: 0.18,
      post_text: "The economy is strong.",
    },
  ],
  narrative_filter_options: {
    topics: ["All", "markets", "trade"],
    policy_buckets: ["All", "economy"],
    stances: ["All", "positive"],
    urgency_bands: ["All", "low", "medium", "high"],
    assets: ["All", "SPY"],
    platforms: ["Truth Social"],
    tracked_scopes: ["All posts", "Trump + tracked accounts", "Tracked accounts only"],
    bucket_fields: [
      { value: "semantic_topic", label: "Topic" },
      { value: "semantic_policy_bucket", label: "Policy bucket" },
    ],
  },
  narrative_metrics: {
    narrative_tagged_posts: 42,
    narrative_sessions: 12,
    cache_hit_rate: 0.9,
    providers_used: 1,
  },
  provider_summary: [
    {
      semantic_provider: "heuristic-fallback",
      posts: 42,
      cache_hit_rate: 0.9,
    },
  ],
  narrative_frequency: [
    {
      session_date: "2026-04-17",
      semantic_topic: "markets",
      post_count: 3,
    },
  ],
  narrative_returns: [
    {
      semantic_topic: "markets",
      avg_next_session_return: 0.01,
      session_count: 4,
    },
  ],
  narrative_asset_heatmap: [
    {
      semantic_topic: "markets",
      asset_symbol: "SPY",
      post_count: 3,
    },
  ],
  narrative_posts: [
    {
      session_date: "2026-04-17",
      semantic_topic: "markets",
      semantic_policy_bucket: "economy",
      author_handle: "realDonaldTrump",
      post_text: "The economy is strong.",
    },
  ],
  narrative_events: [
    {
      trade_date: "2026-04-17",
      post_count: 3,
      primary_topics: "markets",
      next_session_return: 0.01,
    },
  ],
  export_filename: "research-pack-20250120-20260420.zip",
};

const discoveryPayload = {
  ready: true,
  message: "Discovery ranks non-Trump X accounts that mention Trump.",
  source_mode: {
    mode: "truth_plus_x",
    truth_post_count: 42,
    x_post_count: 12,
  },
  latest_ranked_at: "2026-04-17T00:00:00",
  summary: {
    post_count: 54,
    x_candidate_post_count: 12,
    active_account_count: 2,
    latest_ranking_count: 3,
    selected_account_count: 2,
    pinned_account_count: 1,
    suppressed_latest_count: 1,
    override_count: 2,
    pin_override_count: 1,
    suppress_override_count: 1,
  },
  charts: {
    top_discovered_accounts: {
      data: [{ type: "bar", orientation: "h", x: [12, 9], y: ["macroalpha", "policywatch"], name: "Discovery score" }],
      layout: { title: { text: "Top discovered accounts" } },
    },
    ranking_history: {
      data: [{ type: "scatter", mode: "lines+markers", x: ["2026-04-17"], y: [3], name: "Ranked accounts" }],
      layout: { title: { text: "Discovery ranking history" } },
    },
  },
  active_accounts: [
    {
      handle: "macroalpha",
      display_name: "Macro Alpha",
      status: "active",
      discovery_score: 12,
      mention_count: 3,
      provenance: "discovery_auto_include",
    },
    {
      handle: "policywatch",
      display_name: "Policy Watch",
      status: "pinned",
      discovery_score: 9,
      mention_count: 1,
      provenance: "manual_override:pin",
    },
  ],
  latest_rankings: [
    {
      author_handle: "macroalpha",
      author_display_name: "Macro Alpha",
      discovery_score: 12,
      mention_count: 3,
      selected_status: "active",
      suppressed_by_override: false,
    },
    {
      author_handle: "policywatch",
      author_display_name: "Policy Watch",
      discovery_score: 9,
      mention_count: 1,
      selected_status: "pinned",
      suppressed_by_override: false,
    },
    {
      author_handle: "muted",
      author_display_name: "Muted Account",
      discovery_score: 3,
      mention_count: 1,
      selected_status: "excluded",
      suppressed_by_override: true,
    },
  ],
  override_history: [
    {
      handle: "policywatch",
      action: "pin",
      note: "Always inspect",
    },
    {
      handle: "muted",
      action: "suppress",
      note: "Low quality",
    },
  ],
  recent_ranking_history: [
    {
      ranked_at: "2026-04-17T00:00:00",
      author_handle: "macroalpha",
      discovery_score: 12,
    },
  ],
};

const livePayload = {
  configured: true,
  errors: [],
  warnings: [],
  decision: {
    winning_asset: "QQQ",
    decision_source: "eligible",
    winner_score: 0.024,
    eligible_asset_count: 2,
  },
  board: [
    {
      asset_symbol: "QQQ",
      expected_return_score: 0.024,
      confidence: 0.71,
      qualifies: true,
      is_winner: true,
    },
    {
      asset_symbol: "SPY",
      expected_return_score: 0.012,
      confidence: 0.62,
      qualifies: true,
      is_winner: false,
    },
  ],
};

const liveOpsPayload = {
  ...livePayload,
  admin: {
    mode: "public",
    write_requires_unlock: true,
    capture_scope: "stored_data_only",
  },
  current_config: {
    mode: "portfolio_run",
    portfolio_run_id: "run-portfolio-1",
    portfolio_run_name: "Portfolio Alpha",
    deployment_variant: "per_asset_hybrid",
    fallback_mode: "SPY",
  },
  seeded_config: {
    mode: "portfolio_run",
    portfolio_run_id: "run-portfolio-1",
    fallback_mode: "SPY",
  },
  run_options: [
    {
      run_id: "run-portfolio-1",
      run_name: "Portfolio Alpha",
      deployment_variant: "per_asset_hybrid",
      deployment_narrative_feature_mode: "hybrid",
      selected_symbols: "SPY, QQQ",
      transaction_cost_bps: 2,
    },
  ],
  asset_history: [],
  decision_history: [
    {
      generated_at: "2026-04-20T12:00:00Z",
      winning_asset: "QQQ",
      winner_score: 0.024,
    },
  ],
  paper: {
    current_config: {
      paper_portfolio_id: "paper-1",
      portfolio_run_name: "Portfolio Alpha",
      enabled: true,
    },
    active_config: {
      paper_portfolio_id: "paper-1",
      portfolio_run_name: "Portfolio Alpha",
      deployment_variant: "per_asset_hybrid",
      transaction_cost_bps: 2,
      starting_cash: 100000,
      enabled: true,
    },
    portfolios: [
      {
        paper_portfolio_id: "paper-1",
        portfolio_run_name: "Portfolio Alpha",
        deployment_variant: "per_asset_hybrid",
        enabled: true,
      },
    ],
    decision_journal: [
      {
        paper_portfolio_id: "paper-1",
        winning_asset: "QQQ",
        settlement_status: "pending",
      },
    ],
    trade_ledger: [],
    equity_curve: [],
    benchmark_curve: [],
  },
  capture_result: {
    persisted_assets: 2,
    persisted_decisions: 1,
    captured: 1,
    settled: 0,
    performance_persisted: true,
  },
};

const portfoliosPayload = {
  current_config: {
    paper_portfolio_id: "paper-1",
  },
  portfolios: [
    {
      paper_portfolio_id: "paper-1",
      portfolio_run_name: "Portfolio Alpha",
      deployment_variant: "per_asset",
      enabled: true,
    },
  ],
};

const performancePayload = {
  persisted: true,
  summary: {
    overall_severity: "warn",
    total_return: 0.031,
    alpha: 0.014,
    fallback_rate: 0.125,
  },
  diagnostics: [
    {
      scope_kind: "decision_quality",
      scope_key: "paper-1",
      metric_name: "score_outcome_correlation",
      severity: "warn",
      observed_value: -0.1,
      detail: "Correlation is below zero.",
    },
    {
      scope_kind: "portfolio",
      scope_key: "paper-1",
      metric_name: "win_rate",
      severity: "ok",
      observed_value: 0.6,
      detail: "",
    },
  ],
  equity_comparison: [],
  rolling_returns: [],
  score_outcomes: [],
  score_buckets: [],
  winner_distribution: [
    {
      winning_asset: "QQQ",
      decision_count: 7,
    },
    {
      winning_asset: "SPY",
      decision_count: 3,
    },
  ],
  drift: [],
};

async function fulfillJson(page: Page, path: string, payload: unknown) {
  await page.route(`**${path}`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(payload),
    });
  });
}

async function mockApi(page: Page) {
  await fulfillJson(page, "/api/status", statusPayload);
  await fulfillJson(page, "/api/datasets/health", healthPayload);
  await fulfillJson(page, "/api/runs", runsPayload);
  await fulfillJson(page, "/api/runs/run-portfolio-1**", runDetailPayload);
  await fulfillJson(page, "/api/runs/compare**", runComparisonPayload);
  await fulfillJson(page, "/api/research**", researchPayload);
  await fulfillJson(page, "/api/discovery", discoveryPayload);
  await fulfillJson(page, "/api/live/current", livePayload);
  await fulfillJson(page, "/api/live/ops", liveOpsPayload);
  await fulfillJson(page, "/api/admin/session", { token: "admin-token", token_type: "bearer", expires_at: "2026-04-20T20:00:00Z", expires_in_seconds: 43200, mode: "public" });
  await fulfillJson(page, "/api/live/config", liveOpsPayload);
  await fulfillJson(page, "/api/live/capture", liveOpsPayload);
  await fulfillJson(page, "/api/paper/current", liveOpsPayload);
  await fulfillJson(page, "/api/paper/portfolios", portfoliosPayload);
  await fulfillJson(page, "/api/performance/paper-1", performancePayload);
}

test.beforeEach(async ({ page }) => {
  await mockApi(page);
});

test("renders the overview shell from API-backed data", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Web-first decision workbench" })).toBeVisible();
  await expect(page.getByText("API base")).toBeVisible();
  await expect(page.getByText("truth_only (42 Truth, 0 X)")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Recent saved runs" })).toBeVisible();
  await expect(page.getByText("Portfolio Alpha")).toBeVisible();
});

test("shows data health warnings and registry rows", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Data Health/ }).click();

  await expect(page.getByRole("heading", { name: "Freshness, completeness, and anomalies" })).toBeVisible();
  await expect(page.getByText("asset_intraday")).toBeVisible();
  await expect(page.getByText("freshness_hours")).toBeVisible();
  await expect(page.getByText("normalized_posts")).toBeVisible();
});

test("shows the migrated research workspace with narratives and export", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Research/ }).click();

  await expect(page.getByRole("heading", { name: "Sentiment, narratives, and export pack" })).toBeVisible();
  await expect(page.getByText("Truth Social-only mode is active")).toBeVisible();
  await expect(page.getByRole("link", { name: "Export research pack" })).toHaveAttribute(
    "href",
    /\/api\/research\/export/,
  );
  await expect(page.getByRole("heading", { name: "Session table" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "realDonaldTrump" }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Structured narrative inspection" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "markets" }).first()).toBeVisible();
  await expect(page.getByText("heuristic-fallback")).toBeVisible();
});

test("shows the migrated discovery workspace with rankings and overrides", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Discovery/ }).click();

  await expect(page.getByRole("heading", { name: "Tracked account ranking workspace" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Discovery workspace" })).toBeVisible();
  await expect(page.getByText("Manual pin/suppress override writes remain in Streamlit")).toBeVisible();
  await expect(page.getByText("Discovery ranks non-Trump X accounts that mention Trump.", { exact: true })).toBeVisible();
  await expect(page.getByText("X candidate posts")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Top discovered accounts" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Active tracked accounts" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "macroalpha" }).first()).toBeVisible();
  await expect(page.getByRole("cell", { name: "policywatch" }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Latest ranking snapshot" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "suppressed by override" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Override history" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "suppress" })).toBeVisible();
});

test("shows the migrated run explorer with detail and comparison tables", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Run Explorer/ }).click();

  await expect(page.getByRole("heading", { name: "Saved model results and comparisons" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Run Explorer" })).toBeVisible();
  await expect(page.getByLabel("Selected run")).toHaveValue("run-portfolio-1");
  await expect(page.getByRole("heading", { name: "Run summary" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "per_asset_hybrid" }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Variant comparison" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "hybrid" }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Narrative lift" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Feature-family impact" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "semantic" }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Session explainability" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "QQQ" }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Compare saved runs" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "SPY Baseline" })).toBeVisible();
  await expect(page.getByText("run-asset-1: robust score")).toBeVisible();
});

test("shows live ops controls and supports admin actions", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Live Ops/ }).click();

  await expect(page.getByRole("heading", { name: "Live Ops Console" })).toBeVisible();
  await expect(page.getByText("Stored-data capture only")).toBeVisible();
  await expect(page.getByText("Read-only until unlocked")).toBeVisible();
  await page.getByPlaceholder("Admin password").fill("secret");
  await page.getByRole("button", { name: "Unlock admin writes" }).click();
  await expect(page.getByText("Unlocked for this browser session")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Pinned portfolio run" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Save pinned portfolio run" })).toBeEnabled();
  await page.getByRole("button", { name: "Save pinned portfolio run" }).click();
  await page.getByRole("button", { name: "Capture current board" }).click();
  await expect(page.getByRole("cell", { name: "QQQ", exact: true }).first()).toBeVisible();
  await expect(page.getByText("eligible", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Current ranked board" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "SPY", exact: true }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Paper portfolio controls" })).toBeVisible();
  await page.getByRole("button", { name: "Disable paper trading" }).click();
  await expect(page.getByRole("heading", { name: "Recent paper journal" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "pending" })).toBeVisible();
});

test("shows paper performance diagnostics and winner distribution", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Paper \+ Performance/ }).click();

  await expect(page.getByRole("heading", { name: "Portfolio selector" })).toBeVisible();
  await expect(page.getByRole("combobox")).toContainText("paper-1 - Portfolio Alpha");
  await expect(page.getByText("score_outcome_correlation")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Winner distribution" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "decision count" })).toBeVisible();
});
