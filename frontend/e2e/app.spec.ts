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

const datasetAdminPayload = {
  ...healthPayload,
  admin: {
    mode: "public",
    write_requires_unlock: true,
  },
  status: {
    operating_mode: "Truth Social-only",
    source_mode: statusPayload.source_mode,
    state_root: statusPayload.state_root,
    db_path: statusPayload.db_path,
    scheduler_enabled: false,
    missing_core_datasets: [],
    missing_core_dataset_count: 0,
    last_refresh: {
      refresh_mode: "incremental",
      status: "success",
      completed_at: "2026-04-20T12:00:00Z",
    },
    active_job_id: "",
  },
  watchlist_symbols: ["NVDA", "TSLA"],
  asset_universe: [
    {
      symbol: "SPY",
      display_name: "SPDR S&P 500 ETF Trust",
      asset_type: "etf",
      source: "core_etf",
    },
    {
      symbol: "NVDA",
      display_name: "NVDA",
      asset_type: "equity",
      source: "watchlist",
    },
  ],
  source_manifests: [
    {
      source: "truth_archive",
      status: "ok",
      post_count: 42,
    },
  ],
  asset_market_manifest: [
    {
      symbol: "SPY",
      dataset_kind: "daily",
      status: "ok",
      row_count: 300,
    },
  ],
  refresh_jobs: [
    {
      job_id: "dataset-refresh-1",
      refresh_mode: "incremental",
      status: "success",
      normalized_post_count: 42,
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

const modelTrainingPayload = {
  admin: {
    write_requires_unlock: true,
  },
  status: {
    ready: true,
    active_job_id: "",
    readiness_errors: {
      single_asset: [],
      saved_run_portfolio: [],
      joint_portfolio: [],
    },
  },
  defaults: {
    single_asset: {
      run_name: "baseline-research-run",
      target_asset: "SPY",
      feature_version: "v1",
    },
    saved_run_portfolio: {
      run_name: "portfolio-allocator-run",
      fallback_mode: "SPY",
    },
    joint_portfolio: {
      run_name: "joint-portfolio-run",
      feature_version: "asset-v1",
      selected_symbols: ["SPY", "QQQ", "NVDA"],
    },
  },
  asset_options: [
    { symbol: "SPY", label: "SPY - SPDR S&P 500 ETF Trust" },
    { symbol: "QQQ", label: "QQQ - Invesco QQQ Trust" },
  ],
  feature_versions: ["asset-v1"],
  asset_session_symbols: ["SPY", "QQQ", "NVDA"],
  narrative_feature_modes: ["baseline", "narrative_only", "hybrid"],
  model_families: ["ridge", "elastic_net", "hist_gradient_boosting_regressor"],
  topology_variants: ["per_asset", "pooled"],
  run_options: runsPayload.runs,
  asset_model_runs: [
    {
      run_id: "run-asset-1",
      run_name: "SPY Baseline",
      target_asset: "SPY",
      run_type: "asset_model",
      selected_params: {
        threshold: 0.001,
      },
    },
    {
      run_id: "run-asset-qqq",
      run_name: "QQQ Baseline",
      target_asset: "QQQ",
      run_type: "asset_model",
    },
  ],
  recent_jobs: [
    {
      job_id: "model-training-1",
      workflow_mode: "joint_portfolio",
      status: "success",
      run_id: "run-portfolio-1",
      run_name: "Portfolio Alpha",
    },
  ],
};

const modelTrainingStartedPayload = {
  ...modelTrainingPayload,
  job_id: "model-training-2",
  status: {
    ...modelTrainingPayload.status,
    active_job_id: "model-training-2",
  },
  recent_jobs: [
    {
      job_id: "model-training-2",
      workflow_mode: "joint_portfolio",
      status: "success",
      run_id: "run-portfolio-2",
      run_name: "Joint Portfolio Web Test",
      run_type: "portfolio_allocator",
      target_asset: "PORTFOLIO",
      summary: JSON.stringify({
        metrics: {
          total_return: 0.07,
          robust_score: 1.4,
        },
      }),
    },
    ...modelTrainingPayload.recent_jobs,
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

const replayPayload = {
  ready: true,
  message: "",
  selected_run_id: "run-asset-1",
  selected_session_date: "2025-03-03",
  min_history_rows: 20,
  run_options: [
    {
      run_id: "run-asset-1",
      run_name: "SPY Baseline",
      target_asset: "SPY",
      run_type: "asset_model",
      allocator_mode: "",
      robust_score: 1.2,
      total_return: 0.08,
    },
  ],
  sessions: [
    {
      value: "2025-03-03",
      label: "2025-03-03 | posts 3 | prior train rows 20",
      signal_session_date: "2025-03-03T00:00:00",
      post_count: 3,
      history_rows_available: 20,
    },
    {
      value: "2025-03-04",
      label: "2025-03-04 | posts 2 | prior train rows 21",
      signal_session_date: "2025-03-04T00:00:00",
      post_count: 2,
      history_rows_available: 21,
    },
  ],
  summary: {
    saved_run_count: 2,
    asset_model_run_count: 1,
    eligible_session_count: 2,
  },
};

const replaySessionPayload = {
  ready: true,
  message: "",
  run_id: "run-asset-1",
  run_name: "SPY Baseline",
  target_asset: "SPY",
  signal_session_date: "2025-03-03T00:00:00",
  metrics: {
    target_asset: "SPY",
    replay_session: "2025-03-03T00:00:00",
    replay_score: 0.015,
    replay_confidence: 0.72,
    suggested_stance: "LONG SPY NEXT SESSION",
    training_rows_used: 20,
    history_start: "2025-01-01T00:00:00",
    history_end: "2025-02-28T00:00:00",
    actual_next_session_return: 0.004,
    full_history_score: 0.02,
    replay_vs_full_history_drift: -0.005,
  },
  metadata: {
    template_run_id: "run-asset-1",
    future_training_leakage: false,
    full_history_comparison_available: true,
    deployment_params: {
      threshold: 0.001,
      min_post_count: 1,
      account_weight: 1,
    },
  },
  prediction: [
    {
      signal_session_date: "2025-03-03",
      expected_return_score: 0.015,
      prediction_confidence: 0.72,
      suggested_stance: "LONG SPY NEXT SESSION",
      future_training_leakage: false,
    },
  ],
  comparison_rows: [
    { metric: "Replay score", value: 0.015 },
    { metric: "Replay vs full-history drift", value: -0.005 },
  ],
  feature_importance: [{ feature_name: "post_count", coefficient: 0.4, abs_coefficient: 0.4 }],
  feature_contributions: [{ feature_name: "post_count", feature_family: "activity", contribution: 0.01 }],
  post_attribution: [{ author_handle: "realDonaldTrump", post_preview: "Replay market post", post_signal_score: 0.4 }],
  account_attribution: [{ author_handle: "realDonaldTrump", post_count: 1, net_post_signal: 0.4 }],
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

const researchAssetPayload = {
  ready: true,
  message: "",
  source_mode: researchPayload.source_mode,
  filters: researchPayload.filters,
  controls: {
    selected_asset: "NVDA",
    comparison_mode: "normalized",
    benchmark_symbol: "QQQ",
    pre_sessions: 3,
    post_sessions: 5,
    before_minutes: 120,
    after_minutes: 240,
    intraday_session_date: "2026-04-17",
    intraday_anchor_post_id: "truth-1",
  },
  asset_options: [{ symbol: "NVDA", label: "NVDA - NVIDIA", source: "watchlist", is_watchlist: true, has_daily: true }],
  benchmark_options: ["None", "QQQ", "XLK", "XLF", "XLE", "SMH"],
  headline_metrics: {
    sessions_in_range: 12,
    spy_move: 0.04,
    asset_move: 0.12,
    asset_vs_spy_spread: 0.08,
    mapped_post_count: 2,
    asset_session_count: 2,
    event_count: 2,
    intraday_bars: 9,
  },
  charts: {
    overlay: {
      data: [{ type: "scatter", mode: "lines", x: ["2026-04-17"], y: [0.12], name: "NVDA" }],
      layout: { title: { text: "SPY vs. NVDA normalized returns" } },
    },
    event_study: {
      data: [{ type: "scatter", mode: "lines+markers", x: [-1, 0, 1], y: [-0.01, 0, 0.02], name: "NVDA" }],
      layout: { title: { text: "SPY vs. NVDA event study" } },
    },
    intraday: {
      data: [{ type: "scatter", mode: "lines", x: ["2026-04-17T13:35:00Z"], y: [0.01], name: "NVDA" }],
      layout: { title: { text: "SPY vs. NVDA intraday reaction" } },
    },
  },
  asset_session_rows: [{ trade_date: "2026-04-17", post_count: 2, next_session_return: 0.012 }],
  mapped_post_rows: [
    {
      asset_symbol: "NVDA",
      session_date: "2026-04-17",
      author_handle: "realDonaldTrump",
      match_reasons: "semantic_primary_asset",
      post_text: "Semiconductor production is strong.",
    },
  ],
  event_study_rows: [{ symbol: "NVDA", relative_session: 1, avg_relative_return: 0.02, event_count: 2 }],
  intraday_anchor_options: [
    {
      anchor_id: "truth-1",
      session_date: "2026-04-17",
      label: "2026-04-17 09:35 ET | @realDonaldTrump | Semiconductor production is strong.",
      post_timestamp: "2026-04-17T13:35:00Z",
    },
  ],
  intraday_coverage_rows: [
    { symbol: "SPY", bars: 3, covered_dates: "2026-04-17" },
    { symbol: "NVDA", bars: 3, covered_dates: "2026-04-17" },
    { symbol: "QQQ", bars: 3, covered_dates: "2026-04-17" },
  ],
  intraday_window_rows: [{ symbol: "NVDA", timestamp: "2026-04-17T13:35:00Z", normalized_return: 0.01 }],
  empty_states: {
    asset_market: "",
    mapped_posts: "",
    event_study: "",
    intraday: "",
  },
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
  admin: {
    mode: "public",
    writes_enabled: true,
    write_disabled_reason: "",
  },
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
  override_account_options: [
    {
      account_id: "acct-macro",
      handle: "macroalpha",
      display_name: "Macro Alpha",
      source_platform: "X",
      status: "active",
      source: "latest_ranking",
      label: "@macroalpha - Macro Alpha",
    },
    {
      account_id: "acct-policy",
      handle: "policywatch",
      display_name: "Policy Watch",
      source_platform: "X",
      status: "pinned",
      source: "latest_ranking",
      label: "@policywatch - Policy Watch",
    },
    {
      account_id: "acct-muted",
      handle: "muted",
      display_name: "Muted Account",
      source_platform: "X",
      status: "excluded",
      source: "latest_ranking",
      label: "@muted - Muted Account",
    },
  ],
  override_history: [
    {
      override_id: "override-pin",
      account_id: "acct-policy",
      handle: "policywatch",
      action: "pin",
      effective_from: "2026-04-01T00:00:00",
      note: "Always inspect",
    },
    {
      override_id: "override-suppress",
      account_id: "acct-muted",
      handle: "muted",
      action: "suppress",
      effective_from: "2026-04-10T00:00:00",
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
  await fulfillJson(page, "/api/datasets/admin", datasetAdminPayload);
  await fulfillJson(page, "/api/datasets/watchlist", datasetAdminPayload);
  await fulfillJson(page, "/api/datasets/refresh", {
    ...datasetAdminPayload,
    job_id: "dataset-refresh-2",
    status: {
      ...datasetAdminPayload.status,
      active_job_id: "dataset-refresh-2",
    },
    refresh_jobs: [
      ...datasetAdminPayload.refresh_jobs,
      {
        job_id: "dataset-refresh-2",
        refresh_mode: "full",
        status: "success",
        uploaded_file_count: 1,
        normalized_post_count: 43,
      },
    ],
  });
  await fulfillJson(page, "/api/datasets/jobs/dataset-refresh-2", {
    job_id: "dataset-refresh-2",
    found: true,
    job: {
      job_id: "dataset-refresh-2",
      refresh_mode: "full",
      status: "success",
    },
    recent_jobs: datasetAdminPayload.refresh_jobs,
  });
  await fulfillJson(page, "/api/runs", runsPayload);
  await fulfillJson(page, "/api/models/training", modelTrainingPayload);
  await fulfillJson(page, "/api/models/jobs", modelTrainingStartedPayload);
  await fulfillJson(page, "/api/models/jobs/model-training-2", {
    job_id: "model-training-2",
    found: true,
    job: modelTrainingStartedPayload.recent_jobs[0],
    recent_jobs: modelTrainingStartedPayload.recent_jobs,
  });
  await fulfillJson(page, "/api/runs/run-portfolio-1**", runDetailPayload);
  await fulfillJson(page, "/api/runs/compare**", runComparisonPayload);
  await fulfillJson(page, "/api/replay**", replayPayload);
  await fulfillJson(page, "/api/replay/session**", replaySessionPayload);
  await fulfillJson(page, "/api/research**", researchPayload);
  await fulfillJson(page, "/api/research/assets**", researchAssetPayload);
  await fulfillJson(page, "/api/discovery", discoveryPayload);
  await fulfillJson(page, "/api/discovery/overrides", discoveryPayload);
  await fulfillJson(page, "/api/discovery/overrides/**", {
    ...discoveryPayload,
    summary: {
      ...discoveryPayload.summary,
      override_count: 1,
      suppress_override_count: 0,
    },
    override_history: discoveryPayload.override_history.slice(0, 1),
  });
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

test("renders primary navigation with reduced motion enabled", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Web-first decision workbench" })).toBeVisible();
  await expect(page.getByText("Workflow map", { exact: true })).toBeVisible();
  await expect(page.getByText("Explore", { exact: true })).toBeVisible();
  await expect(page.getByText("Build", { exact: true })).toBeVisible();
  await expect(page.getByText("Operate", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Help: Workflow map" }).focus();
  await expect(page.getByText(/Follow the flow from data and research/)).toBeVisible();
  await page.getByRole("button", { name: /Research/ }).click();
  await expect(page.getByRole("heading", { name: "Sentiment, narratives, and export pack" })).toBeVisible();
  await expect(page.getByText(/Use this page to inspect filtered Trump Truth Social/).first()).toBeVisible();
  await expect(page.getByRole("link", { name: "Export research pack" })).toBeVisible();
});

test("shows data admin controls, health rows, and refresh jobs", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Data Admin/ }).click();

  await expect(page.getByRole("heading", { name: "Refresh jobs, watchlist, and data health" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Data Admin Console" })).toBeVisible();
  await expect(page.getByText("Read-only until unlocked")).toBeVisible();
  await page.getByPlaceholder("Admin password").fill("secret");
  await page.getByRole("button", { name: "Unlock admin writes" }).click();
  await expect(page.getByText("Unlocked for this browser session")).toBeVisible();
  await page.getByLabel("Watchlist symbols").fill("NVDA, TSLA, XOM");
  await page.getByRole("button", { name: "Save watchlist" }).click();
  await page.getByLabel("Remote X / mention CSV URL").fill("https://example.invalid/mentions.csv");
  await page.locator('input[type="file"]').setInputFiles({
    name: "mentions.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("author,text\nmacro,hello\n"),
  });
  await page.getByRole("button", { name: "Start refresh job" }).click();
  await expect(page.getByRole("cell", { name: "dataset-refresh-2" })).toBeVisible();
  await expect(page.getByText("asset_intraday")).toBeVisible();
  await expect(page.getByText("freshness_hours")).toBeVisible();
  await expect(page.getByText("normalized_posts")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Source manifest" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "truth_archive" })).toBeVisible();
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
  await expect(page.getByRole("heading", { name: "Multi-asset comparison, event study, and intraday reaction" })).toBeVisible();
  await expect(page.locator("label", { hasText: "Selected asset" }).locator("select")).toHaveValue("NVDA");
  await expect(page.getByLabel("ETF baseline")).toHaveValue("QQQ");
  await expect(page.getByText("Mapped posts", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "SPY vs. selected asset" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Event study", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Intraday reaction", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Mapped posts for selected asset" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "semantic_primary_asset" }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Intraday coverage" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "NVDA" }).first()).toBeVisible();
});

test("shows the migrated discovery workspace with rankings and overrides", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Discovery/ }).click();

  await expect(page.getByRole("heading", { name: "Tracked account ranking workspace" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Discovery workspace" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Discovery admin overrides" })).toBeVisible();
  await expect(page.getByText("Admins can now pin or suppress accounts from this")).toBeVisible();
  await expect(page.getByText("Read-only until unlocked")).toBeVisible();
  await expect(page.getByRole("button", { name: "Save discovery override" })).toBeDisabled();
  await page.getByPlaceholder("Admin password").fill("secret");
  await page.getByRole("button", { name: "Unlock admin writes" }).click();
  await expect(page.getByText("Unlocked for this browser session")).toBeVisible();
  await page.getByLabel("Override account").selectOption("acct-muted");
  await page.getByLabel("Override action").selectOption("suppress");
  await page.getByPlaceholder("Optional rationale").fill("Noisy duplicate account");
  await page.getByRole("button", { name: "Save discovery override" }).click();
  await page.getByLabel("Override to delete").selectOption("override-suppress");
  await page.getByRole("button", { name: "Delete selected override" }).click();
  await expect(page.getByText("X candidate posts")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Top discovered accounts" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Active tracked accounts" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "macroalpha" }).first()).toBeVisible();
  await expect(page.getByRole("cell", { name: "policywatch" }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Latest ranking snapshot" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "suppressed by override" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Override history" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "pin", exact: true })).toBeVisible();
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

test("shows historical replay metrics and attribution tables", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Replay/ }).click();

  await expect(page.getByRole("heading", { name: "Historical signal reconstruction" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Historical Replay Workspace" })).toBeVisible();
  await expect(page.getByLabel("Replay template run")).toHaveValue("run-asset-1");
  await expect(page.getByLabel("Historical signal session")).toHaveValue("2025-03-03");
  await expect(page.getByText("LONG SPY NEXT SESSION").first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Full-history comparison" })).toBeVisible();
  await expect(page.getByText("Drift context only")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Replay summary" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "Replay vs full-history drift" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Replay feature importance" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "post_count" }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Feature contributions" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Post attribution" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "realDonaldTrump" }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Account attribution" })).toBeVisible();
});

test("shows model training forms and submits an admin job", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Model Training/ }).click();

  await expect(page.getByRole("heading", { name: "Model Training Job Console" })).toBeVisible();
  await expect(page.getByText("Read-only until unlocked")).toBeVisible();
  await expect(page.getByRole("button", { name: "Single Asset" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Saved-Run Portfolio" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Joint Portfolio" })).toBeVisible();
  await page.getByPlaceholder("Admin password").fill("secret");
  await page.getByRole("button", { name: "Unlock admin writes" }).click();
  await expect(page.getByText("Unlocked for this browser session")).toBeVisible();
  await page.getByRole("button", { name: "Saved-Run Portfolio" }).click();
  await expect(page.getByRole("heading", { name: "Saved-Run Portfolio configuration" })).toBeVisible();
  await page.getByRole("button", { name: "Single Asset" }).click();
  await expect(page.getByRole("heading", { name: "Single Asset configuration" })).toBeVisible();
  await page.getByRole("button", { name: "Joint Portfolio" }).click();
  await expect(page.getByRole("heading", { name: "Joint Portfolio configuration" })).toBeVisible();
  await page.getByRole("button", { name: "Start model training job" }).click();
  await expect(page.getByRole("heading", { name: "Latest training result" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Inspect in Run Explorer" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Configure in Live Ops" })).toBeEnabled();
  await expect(page.getByRole("cell", { name: "model-training-2" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "run-portfolio-2" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Saved asset-model run options" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "QQQ Baseline" })).toBeVisible();
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
