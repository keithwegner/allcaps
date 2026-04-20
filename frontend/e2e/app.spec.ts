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
  count: 1,
  runs: [
    {
      run_id: "run-portfolio-1",
      run_name: "Portfolio Alpha",
      run_type: "portfolio_allocator",
      allocator_mode: "joint_model",
      target_asset: "PORTFOLIO",
    },
  ],
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
  await fulfillJson(page, "/api/research**", researchPayload);
  await fulfillJson(page, "/api/live/current", livePayload);
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

test("shows the current live decision and candidate board", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Live Decision/ }).click();

  await expect(page.getByRole("cell", { name: "QQQ" })).toBeVisible();
  await expect(page.getByText("eligible", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Current ranked board" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "SPY" })).toBeVisible();
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
