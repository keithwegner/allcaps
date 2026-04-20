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
