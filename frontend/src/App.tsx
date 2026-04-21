import { useEffect, useMemo, useState } from "react";
import type { ComponentType } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import createPlotlyComponentModule from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { Data, Frame, Layout } from "plotly.js";
import { api, type DatasetAdminPayload, type LiveOpsPayload, type PlotlyFigure, type RecordRow, type ResearchFilters } from "./api";

type PlotComponentFactory = (plotly: unknown) => ComponentType<Record<string, unknown>>;
const createPlotlyComponent = (
  (createPlotlyComponentModule as { default?: PlotComponentFactory }).default ?? createPlotlyComponentModule
) as PlotComponentFactory;
const Plot = createPlotlyComponent(Plotly);

type PageKey = "overview" | "research" | "discovery" | "runs" | "data" | "live" | "paper";

const pages: Array<{ key: PageKey; label: string; deck: string }> = [
  { key: "overview", label: "Overview", deck: "API status and migration posture" },
  { key: "research", label: "Research", deck: "Sentiment, narratives, and export pack" },
  { key: "discovery", label: "Discovery", deck: "Tracked account ranking workspace" },
  { key: "runs", label: "Run Explorer", deck: "Saved model results and comparisons" },
  { key: "data", label: "Data Admin", deck: "Refresh jobs, watchlist, and data health" },
  { key: "live", label: "Live Ops", deck: "Operate deployed portfolio" },
  { key: "paper", label: "Paper + Performance", deck: "Portfolio audit and drift" },
];

function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "n/a";
  }
  if (typeof value === "number") {
    if (Math.abs(value) < 1 && value !== 0) {
      return `${(value * 100).toFixed(2)}%`;
    }
    return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4);
  }
  return String(value);
}

function StatusPill({ label, tone = "neutral" }: { label: string; tone?: "neutral" | "ok" | "warn" | "severe" }) {
  return <span className={`status-pill status-pill--${tone}`}>{label}</span>;
}

function LoadingBlock({ label }: { label: string }) {
  return <div className="loading-block">{label}</div>;
}

function ErrorBlock({ error }: { error: unknown }) {
  return <div className="error-block">{error instanceof Error ? error.message : "Unable to load API data."}</div>;
}

function MetricCard({ label, value, caption }: { label: string; value: unknown; caption?: string }) {
  return (
    <article className="metric-card">
      <span>{label}</span>
      <strong>{formatValue(value)}</strong>
      {caption ? <small>{caption}</small> : null}
    </article>
  );
}

function selectedOptions(options: HTMLCollectionOf<HTMLOptionElement>): string[] {
  return Array.from(options)
    .filter((option) => option.selected)
    .map((option) => option.value);
}

function PlotlyChart({ figure, title }: { figure?: PlotlyFigure; title: string }) {
  if (!figure) {
    return <div className="empty-state">No chart data returned yet.</div>;
  }
  return (
    <div className="chart-frame" aria-label={title}>
      <Plot
        data={(figure.data ?? []) as Data[]}
        layout={
          {
            autosize: true,
            ...(figure.layout ?? {}),
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
          } as Partial<Layout>
        }
        frames={(figure.frames ?? []) as Frame[]}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
      />
    </div>
  );
}

function DataTable({ rows, emptyLabel = "No rows returned yet." }: { rows: RecordRow[]; emptyLabel?: string }) {
  const columns = useMemo(() => {
    const keys = new Set<string>();
    rows.slice(0, 12).forEach((row) => Object.keys(row).forEach((key) => keys.add(key)));
    return Array.from(keys).slice(0, 10);
  }, [rows]);

  if (!rows.length) {
    return <div className="empty-state">{emptyLabel}</div>;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column.replaceAll("_", " ")}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 25).map((row, index) => (
            <tr key={`${index}-${String(row[columns[0]] ?? "")}`}>
              {columns.map((column) => (
                <td key={column}>{formatValue(row[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function parseSymbolList(value: string): string[] {
  return value
    .split(/[\s,]+/)
    .map((item) => item.trim().toUpperCase())
    .filter(Boolean);
}

function ResearchPage() {
  const [filters, setFilters] = useState<ResearchFilters>({});
  const research = useQuery({ queryKey: ["research", filters], queryFn: () => api.research(filters) });

  const payload = research.data;
  const currentFilters = { ...(payload?.filters ?? {}), ...filters };
  const options = payload?.narrative_filter_options ?? {};
  const updateFilters = (next: ResearchFilters) => setFilters((previous) => ({ ...previous, ...next }));

  if (research.isLoading) {
    return <LoadingBlock label="Loading research workspace..." />;
  }
  if (research.error) {
    return <ErrorBlock error={research.error} />;
  }

  return (
    <section className="page-grid">
      <article className="panel panel--wide">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Read-only migration slice</p>
            <h2>Research workspace</h2>
          </div>
          <StatusPill label={payload?.source_mode.mode === "truth_only" ? "Truth Social-only" : "Research API"} tone="ok" />
        </div>
        {payload?.source_mode.mode === "truth_only" ? (
          <p>
            Truth Social-only mode is active. The default scope is Donald Trump-authored Truth Social posts; use the platform
            and author controls below to verify or intentionally broaden the slice.
          </p>
        ) : null}
        {!payload?.ready ? <div className="empty-state">{payload?.message ?? "Research data is not ready yet."}</div> : null}
        <div className="filter-grid">
          <label>
            Start date
            <input
              type="date"
              value={currentFilters.date_start ?? ""}
              onChange={(event) => updateFilters({ date_start: event.target.value })}
            />
          </label>
          <label>
            End date
            <input
              type="date"
              value={currentFilters.date_end ?? ""}
              onChange={(event) => updateFilters({ date_end: event.target.value })}
            />
          </label>
          <label>
            Platforms
            <select
              multiple
              value={currentFilters.platforms ?? []}
              onChange={(event) => updateFilters({ platforms: selectedOptions(event.currentTarget.selectedOptions) })}
            >
              <option value="Truth Social">Truth Social</option>
              <option value="X">X</option>
            </select>
          </label>
          <label>
            Keyword
            <input
              type="search"
              value={currentFilters.keyword ?? ""}
              onChange={(event) => updateFilters({ keyword: event.target.value })}
              placeholder="Optional exact text filter"
            />
          </label>
          <label className="checkbox-field">
            <input
              type="checkbox"
              checked={Boolean(currentFilters.trump_authored_only)}
              onChange={(event) => updateFilters({ trump_authored_only: event.target.checked })}
            />
            Trump-authored only
          </label>
          <label className="checkbox-field">
            <input
              type="checkbox"
              checked={Boolean(currentFilters.include_reshares)}
              onChange={(event) => updateFilters({ include_reshares: event.target.checked })}
            />
            Include reshares
          </label>
          <label className="checkbox-field">
            <input
              type="checkbox"
              checked={Boolean(currentFilters.tracked_only)}
              onChange={(event) => updateFilters({ tracked_only: event.target.checked })}
            />
            Trump + tracked only
          </label>
          <label className="checkbox-field">
            <input
              type="checkbox"
              checked={currentFilters.scale_markers !== false}
              onChange={(event) => updateFilters({ scale_markers: event.target.checked })}
            />
            Scale activity
          </label>
        </div>
      </article>

      <div className="metric-grid">
        <MetricCard label="Sessions with posts" value={payload?.headline_metrics.sessions_with_posts ?? 0} />
        <MetricCard label="Posts in view" value={payload?.headline_metrics.posts_in_view ?? 0} />
        <MetricCard label="Truth posts" value={payload?.headline_metrics.truth_posts ?? 0} />
        <MetricCard label="Mean sentiment" value={payload?.headline_metrics.mean_sentiment ?? 0} />
      </div>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Social activity vs. market baseline</h2>
          <a className="button-link" href={api.researchExportUrl(currentFilters)} download={payload?.export_filename ?? "research-pack.zip"}>
            Export research pack
          </a>
        </div>
        <PlotlyChart figure={payload?.charts.social_activity} title="Social activity vs. market baseline" />
      </article>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Session table</h2>
        </div>
        <DataTable rows={payload?.session_rows ?? []} />
      </article>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Underlying posts</h2>
        </div>
        <DataTable rows={payload?.post_rows ?? []} />
      </article>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Narrative Lab</p>
            <h2>Structured narrative inspection</h2>
          </div>
          <StatusPill label={`${payload?.narrative_metrics.narrative_tagged_posts ?? 0} posts`} />
        </div>
        <div className="filter-grid filter-grid--compact">
          <label>
            Topic
            <select value={currentFilters.narrative_topic ?? "All"} onChange={(event) => updateFilters({ narrative_topic: event.target.value })}>
              {(options.topics ?? ["All"]).map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label>
            Policy bucket
            <select value={currentFilters.narrative_policy ?? "All"} onChange={(event) => updateFilters({ narrative_policy: event.target.value })}>
              {(options.policy_buckets ?? ["All"]).map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label>
            Stance
            <select value={currentFilters.narrative_stance ?? "All"} onChange={(event) => updateFilters({ narrative_stance: event.target.value })}>
              {(options.stances ?? ["All"]).map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label>
            Urgency
            <select value={currentFilters.narrative_urgency ?? "All"} onChange={(event) => updateFilters({ narrative_urgency: event.target.value })}>
              {(options.urgency_bands ?? ["All", "low", "medium", "high"]).map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label>
            Asset target
            <select value={currentFilters.narrative_asset ?? "All"} onChange={(event) => updateFilters({ narrative_asset: event.target.value })}>
              {(options.assets ?? ["All"]).map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label>
            Return bucket
            <select
              value={currentFilters.narrative_bucket_field ?? "semantic_topic"}
              onChange={(event) => updateFilters({ narrative_bucket_field: event.target.value })}
            >
              {(options.bucket_fields ?? [{ value: "semantic_topic", label: "Topic" }]).map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Narrative platforms
            <select
              multiple
              value={currentFilters.narrative_platforms ?? options.platforms ?? []}
              onChange={(event) => updateFilters({ narrative_platforms: selectedOptions(event.currentTarget.selectedOptions) })}
            >
              {(options.platforms ?? []).map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label>
            Tracked scope
            <select
              value={currentFilters.narrative_tracked_scope ?? "All posts"}
              onChange={(event) => updateFilters({ narrative_tracked_scope: event.target.value })}
            >
              {(options.tracked_scopes ?? ["All posts"]).map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="metric-grid metric-grid--small">
          <MetricCard label="Narrative sessions" value={payload?.narrative_metrics.narrative_sessions ?? 0} />
          <MetricCard label="Cache hit rate" value={payload?.narrative_metrics.cache_hit_rate ?? 0} />
          <MetricCard label="Providers used" value={payload?.narrative_metrics.providers_used ?? 0} />
          <MetricCard label="Rows in heatmap" value={payload?.narrative_asset_heatmap.length ?? 0} />
        </div>
      </article>

      <article className="panel panel--wide chart-grid">
        <div>
          <h2>Narrative frequency</h2>
          <PlotlyChart figure={payload?.charts.narrative_frequency} title="Narrative frequency" />
        </div>
        <div>
          <h2>Next-session return by narrative</h2>
          <PlotlyChart figure={payload?.charts.narrative_returns} title="Next-session return by narrative" />
        </div>
      </article>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Asset-by-narrative heatmap</h2>
        </div>
        <PlotlyChart figure={payload?.charts.narrative_asset_heatmap} title="Asset-by-narrative heatmap" />
      </article>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Top narrative posts</h2>
        </div>
        <DataTable rows={payload?.narrative_posts ?? []} />
      </article>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Selected-narrative event table</h2>
        </div>
        <DataTable rows={payload?.narrative_events ?? []} />
      </article>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Provider and cache indicators</h2>
        </div>
        <DataTable rows={payload?.provider_summary ?? []} />
      </article>
    </section>
  );
}

function DiscoveryPage() {
  const discovery = useQuery({ queryKey: ["discovery"], queryFn: api.discovery });

  if (discovery.isLoading) {
    return <LoadingBlock label="Loading discovery workspace..." />;
  }
  if (discovery.error) {
    return <ErrorBlock error={discovery.error} />;
  }

  const payload = discovery.data;
  const summary = payload?.summary ?? {};
  return (
    <section className="page-grid">
      <article className="panel panel--wide">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Read-only migration slice</p>
            <h2>Discovery workspace</h2>
          </div>
          <StatusPill label={payload?.ready ? "Rankings available" : "Guidance"} tone={payload?.ready ? "ok" : "warn"} />
        </div>
        <p>
          Discovery ranks non-Trump X accounts that mention Trump. Manual pin/suppress override writes remain in Streamlit
          for this migration slice.
        </p>
        {payload?.message ? <div className="empty-state">{payload.message}</div> : null}
      </article>

      <div className="metric-grid">
        <MetricCard label="X candidate posts" value={summary.x_candidate_post_count ?? 0} />
        <MetricCard label="Active accounts" value={summary.active_account_count ?? 0} />
        <MetricCard label="Latest rankings" value={summary.latest_ranking_count ?? 0} />
        <MetricCard label="Overrides" value={summary.override_count ?? 0} />
      </div>

      <article className="panel panel--wide chart-grid">
        <div>
          <h2>Top discovered accounts</h2>
          <PlotlyChart figure={payload?.charts.top_discovered_accounts} title="Top discovered accounts" />
        </div>
        <div>
          <h2>Ranking history</h2>
          <PlotlyChart figure={payload?.charts.ranking_history} title="Discovery ranking history" />
        </div>
      </article>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Active tracked accounts</h2>
          <StatusPill label={`${summary.active_account_count ?? 0} active`} />
        </div>
        <DataTable rows={payload?.active_accounts ?? []} />
      </article>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Latest ranking snapshot</h2>
          <StatusPill label={payload?.latest_ranked_at ? `Ranked ${payload.latest_ranked_at}` : "No snapshot"} />
        </div>
        <DataTable rows={payload?.latest_rankings ?? []} />
      </article>

      <article className="panel panel--wide chart-grid">
        <div>
          <h2>Override history</h2>
          <DataTable rows={payload?.override_history ?? []} />
        </div>
        <div>
          <h2>Recent ranking history</h2>
          <DataTable rows={payload?.recent_ranking_history ?? []} />
        </div>
      </article>
    </section>
  );
}

function RunExplorerPage() {
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.runs });
  const [selectedRunId, setSelectedRunId] = useState("");
  const [runTypeFilter, setRunTypeFilter] = useState("All");
  const [allocatorFilter, setAllocatorFilter] = useState("All");
  const [assetFilter, setAssetFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [variantName, setVariantName] = useState("");
  const [sessionDate, setSessionDate] = useState("");
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [baseRunId, setBaseRunId] = useState("");

  const allRuns = runs.data?.runs ?? [];
  const runTypes = useMemo(() => ["All", ...Array.from(new Set(allRuns.map((row) => String(row.run_type ?? "asset_model"))))], [allRuns]);
  const allocatorModes = useMemo(() => ["All", ...Array.from(new Set(allRuns.map((row) => String(row.allocator_mode ?? "")).filter(Boolean)))], [allRuns]);
  const targetAssets = useMemo(() => ["All", ...Array.from(new Set(allRuns.map((row) => String(row.target_asset ?? "SPY"))))], [allRuns]);
  const filteredRuns = useMemo(() => {
    const query = search.trim().toLowerCase();
    return allRuns.filter((row) => {
      if (runTypeFilter !== "All" && String(row.run_type ?? "asset_model") !== runTypeFilter) {
        return false;
      }
      if (allocatorFilter !== "All" && String(row.allocator_mode ?? "") !== allocatorFilter) {
        return false;
      }
      if (assetFilter !== "All" && String(row.target_asset ?? "SPY") !== assetFilter) {
        return false;
      }
      if (!query) {
        return true;
      }
      return `${row.run_id ?? ""} ${row.run_name ?? ""} ${row.target_asset ?? ""}`.toLowerCase().includes(query);
    });
  }, [allRuns, allocatorFilter, assetFilter, runTypeFilter, search]);

  const activeRunId = selectedRunId || String(filteredRuns[0]?.run_id ?? allRuns[0]?.run_id ?? "");
  const detail = useQuery({
    queryKey: ["run-detail", activeRunId, variantName, sessionDate],
    queryFn: () => api.runDetail(activeRunId, { variantName, sessionDate }),
    enabled: Boolean(activeRunId),
  });

  const defaultCompareIds = useMemo(() => {
    const ids = allRuns.map((row) => String(row.run_id ?? "")).filter(Boolean);
    if (activeRunId && !ids.includes(activeRunId)) {
      return [activeRunId];
    }
    return ids.slice(0, Math.min(2, ids.length));
  }, [activeRunId, allRuns]);
  const comparisonRunIds = compareIds.length ? compareIds : defaultCompareIds;
  const activeBaseRunId = baseRunId || comparisonRunIds[0] || "";
  const comparison = useQuery({
    queryKey: ["run-comparison", comparisonRunIds, activeBaseRunId],
    queryFn: () => api.runComparison(comparisonRunIds, activeBaseRunId),
    enabled: comparisonRunIds.length > 0,
  });

  if (runs.isLoading) {
    return <LoadingBlock label="Loading saved runs..." />;
  }
  if (runs.error) {
    return <ErrorBlock error={runs.error} />;
  }
  if (!allRuns.length) {
    return <div className="empty-state">No saved runs have been created yet. Train models in Streamlit first, then return here.</div>;
  }

  const payload = detail.data;
  const variantRows = payload?.tables.variant_summary ?? [];
  const sessionOptions = payload?.session_options ?? [];

  return (
    <section className="page-grid">
      <article className="panel panel--wide">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Read-only migration slice</p>
            <h2>Run Explorer</h2>
          </div>
          <StatusPill label={`${filteredRuns.length} visible runs`} tone="ok" />
        </div>
        <div className="filter-grid filter-grid--compact">
          <label>
            Run type
            <select value={runTypeFilter} onChange={(event) => setRunTypeFilter(event.target.value)}>
              {runTypes.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label>
            Allocator mode
            <select value={allocatorFilter} onChange={(event) => setAllocatorFilter(event.target.value)}>
              {allocatorModes.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label>
            Target asset
            <select value={assetFilter} onChange={(event) => setAssetFilter(event.target.value)}>
              {targetAssets.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label>
            Search
            <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Run name, id, or asset" />
          </label>
          <label>
            Selected run
            <select
              value={activeRunId}
              onChange={(event) => {
                setSelectedRunId(event.target.value);
                setVariantName("");
                setSessionDate("");
              }}
            >
              {filteredRuns.map((row) => (
                <option key={String(row.run_id)} value={String(row.run_id)}>
                  {String(row.run_name ?? row.run_id)} ({String(row.target_asset ?? "SPY")})
                </option>
              ))}
            </select>
          </label>
          <label>
            Variant
            <select value={variantName} onChange={(event) => setVariantName(event.target.value)}>
              <option value="">Deployment default</option>
              {variantRows.map((row) => (
                <option key={String(row.variant_name)} value={String(row.variant_name)}>
                  {String(row.variant_name)} {row.deployment_winner ? "(winner)" : ""}
                </option>
              ))}
            </select>
          </label>
          <label>
            Session inspector
            <select value={sessionDate} onChange={(event) => setSessionDate(event.target.value)}>
              <option value="">Latest session</option>
              {sessionOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      </article>

      {detail.isLoading ? <LoadingBlock label="Loading run detail..." /> : null}
      {detail.error ? <ErrorBlock error={detail.error} /> : null}
      {payload && !payload.found ? <div className="empty-state">{payload.errors.join(" ")}</div> : null}
      {payload?.found ? (
        <>
          <div className="metric-grid">
            <MetricCard label="Total return" value={payload.metrics.total_return ?? 0} />
            <MetricCard label="Robust score" value={payload.metrics.robust_score ?? 0} />
            <MetricCard label="Max drawdown" value={payload.metrics.max_drawdown ?? 0} />
            <MetricCard label="Feature count" value={payload.model_artifact.feature_count ?? 0} />
          </div>

          <article className="panel panel--wide">
            <div className="panel-heading">
              <h2>Run summary</h2>
              <StatusPill label={String(payload.settings.run_type ?? "asset_model")} />
            </div>
            <dl className="detail-list">
              <div>
                <dt>Run</dt>
                <dd>{String(payload.run.run_name ?? payload.run_id)}</dd>
              </div>
              <div>
                <dt>Target asset</dt>
                <dd>{String(payload.settings.target_asset ?? "SPY")}</dd>
              </div>
              <div>
                <dt>Deployment variant</dt>
                <dd>{String(payload.settings.deployment_variant ?? "n/a")}</dd>
              </div>
              <div>
                <dt>Narrative mode</dt>
                <dd>{String(payload.settings.deployment_narrative_feature_mode ?? "n/a")}</dd>
              </div>
              <div>
                <dt>Allocator</dt>
                <dd>{String(payload.settings.allocator_mode ?? "n/a")}</dd>
              </div>
              <div>
                <dt>Rows loaded</dt>
                <dd>{formatValue(payload.row_counts.trades)} trades</dd>
              </div>
            </dl>
          </article>

          <article className="panel panel--wide chart-grid">
            <div>
              <h2>Equity curve</h2>
              <PlotlyChart figure={payload.charts.equity} title="Run equity curve" />
            </div>
            <div>
              <h2>Diagnostics chart</h2>
              <PlotlyChart figure={payload.charts.diagnostics} title="Run diagnostics chart" />
            </div>
          </article>

          <article className="panel panel--wide">
            <div className="panel-heading">
              <h2>Benchmark curves</h2>
            </div>
            <PlotlyChart figure={payload.charts.benchmarks} title="Benchmark curves" />
          </article>

          <article className="panel panel--wide chart-grid">
            <div>
              <h2>Variant comparison</h2>
              <DataTable rows={payload.tables.variant_summary ?? []} />
            </div>
            <div>
              <h2>Narrative lift</h2>
              <DataTable rows={payload.tables.narrative_lift ?? []} />
            </div>
          </article>

          <article className="panel panel--wide chart-grid">
            <div>
              <h2>Feature-family impact</h2>
              <DataTable rows={payload.tables.feature_family_summary ?? []} />
            </div>
            <div>
              <h2>Feature importance</h2>
              <DataTable rows={payload.tables.feature_importance ?? []} />
            </div>
          </article>

          <article className="panel panel--wide chart-grid">
            <div>
              <h2>Session explainability</h2>
              <DataTable rows={payload.selected_session.decision?.length ? payload.selected_session.decision : payload.selected_session.prediction ?? []} />
            </div>
            <div>
              <h2>Session candidates</h2>
              <DataTable rows={payload.selected_session.candidates ?? []} />
            </div>
          </article>

          <article className="panel panel--wide chart-grid">
            <div>
              <h2>Feature contributions</h2>
              <DataTable rows={payload.selected_session.feature_contributions ?? []} />
            </div>
            <div>
              <h2>Post attribution</h2>
              <DataTable rows={payload.selected_session.post_attribution ?? []} />
            </div>
          </article>

          <article className="panel panel--wide chart-grid">
            <div>
              <h2>Diagnostics rows</h2>
              <DataTable rows={payload.tables.diagnostics ?? []} />
            </div>
            <div>
              <h2>Recent trades</h2>
              <DataTable rows={payload.tables.trades ?? []} />
            </div>
          </article>
        </>
      ) : null}

      <article className="panel panel--wide">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Comparison Workspace</p>
            <h2>Compare saved runs</h2>
          </div>
          <StatusPill label={comparison.data?.ready ? "Ready" : "Select 2+ runs"} />
        </div>
        <div className="filter-grid filter-grid--compact">
          <label>
            Compare runs
            <select
              multiple
              value={comparisonRunIds}
              onChange={(event) => setCompareIds(selectedOptions(event.currentTarget.selectedOptions))}
            >
              {allRuns.map((row) => (
                <option key={String(row.run_id)} value={String(row.run_id)}>
                  {String(row.run_name ?? row.run_id)}
                </option>
              ))}
            </select>
          </label>
          <label>
            Base run
            <select value={activeBaseRunId} onChange={(event) => setBaseRunId(event.target.value)}>
              {comparisonRunIds.map((runId) => (
                <option key={runId} value={runId}>
                  {runId}
                </option>
              ))}
            </select>
          </label>
        </div>
        {comparison.isLoading ? <LoadingBlock label="Loading run comparison..." /> : null}
        {comparison.error ? <ErrorBlock error={comparison.error} /> : null}
        {comparison.data ? (
          <>
            <PlotlyChart figure={comparison.data.charts.equity} title="Run comparison equity" />
            <div className="chart-grid">
              <div>
                <h2>Scorecard</h2>
                <DataTable rows={comparison.data.scorecard} />
              </div>
              <div>
                <h2>Setting diffs</h2>
                <DataTable rows={comparison.data.setting_diffs} />
              </div>
              <div>
                <h2>Feature diffs</h2>
                <DataTable rows={comparison.data.feature_diffs} />
              </div>
              <div>
                <h2>Benchmark deltas</h2>
                <DataTable rows={comparison.data.benchmark_deltas} />
              </div>
            </div>
            <div className="note-list">
              {(comparison.data.change_notes.length ? comparison.data.change_notes : ["Choose at least one non-base run to summarize what changed."]).map((note) => (
                <p key={note}>{note}</p>
              ))}
            </div>
          </>
        ) : null}
      </article>
    </section>
  );
}

function OverviewPage() {
  const status = useQuery({ queryKey: ["status"], queryFn: api.status });
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.runs });

  if (status.isLoading || runs.isLoading) {
    return <LoadingBlock label="Loading API status..." />;
  }
  if (status.error || runs.error) {
    return <ErrorBlock error={status.error ?? runs.error} />;
  }

  const payload = status.data;
  return (
    <section className="page-grid">
      <div className="metric-grid">
        <MetricCard label="App mode" value={payload?.mode} />
        <MetricCard label="Datasets tracked" value={payload?.dataset_count ?? 0} />
        <MetricCard label="Missing core datasets" value={payload?.missing_core_datasets.length ?? 0} />
        <MetricCard label="Saved runs" value={runs.data?.count ?? 0} />
      </div>
      <article className="panel panel--wide">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Migration checkpoint</p>
            <h2>Streamlit remains live while the web app grows beside it.</h2>
          </div>
          <StatusPill label="Read-only API" tone="ok" />
        </div>
        <p>
          This React shell is the first web-first surface. It talks to the FastAPI backend,
          which reuses the existing DuckDB, Parquet, modeling, live-monitor, paper-trading,
          and observability services.
        </p>
        <dl className="detail-list">
          <div>
            <dt>API base</dt>
            <dd>{api.baseUrl}</dd>
          </div>
          <div>
            <dt>State root</dt>
            <dd>{payload?.state_root}</dd>
          </div>
          <div>
            <dt>Source mode</dt>
            <dd>
              {payload?.source_mode.mode} ({payload?.source_mode.truth_post_count} Truth,{" "}
              {payload?.source_mode.x_post_count} X)
            </dd>
          </div>
        </dl>
      </article>
      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Recent saved runs</h2>
        </div>
        <DataTable rows={runs.data?.runs ?? []} />
      </article>
    </section>
  );
}

function DataAdminPage() {
  const queryClient = useQueryClient();
  const admin = useQuery({ queryKey: ["dataset-admin"], queryFn: api.datasetAdmin, refetchInterval: 10_000 });
  const [adminToken, setAdminToken] = useState(() =>
    typeof window === "undefined" ? "" : window.sessionStorage.getItem("allcaps_admin_token") ?? "",
  );
  const [password, setPassword] = useState("");
  const [watchlistText, setWatchlistText] = useState("");
  const [refreshMode, setRefreshMode] = useState<"bootstrap" | "full" | "incremental">("incremental");
  const [remoteUrl, setRemoteUrl] = useState("");
  const [uploadFiles, setUploadFiles] = useState<FileList | null>(null);

  const applyDatasetPayload = (payload: DatasetAdminPayload) => {
    queryClient.setQueryData(["dataset-admin"], payload);
  };
  const unlock = useMutation({
    mutationFn: () => api.adminSession(password),
    onSuccess: (payload) => {
      setAdminToken(payload.token);
      if (typeof window !== "undefined") {
        window.sessionStorage.setItem("allcaps_admin_token", payload.token);
      }
      setPassword("");
    },
  });
  const saveWatchlist = useMutation({
    mutationFn: () => api.saveWatchlist(parseSymbolList(watchlistText), false, adminToken),
    onSuccess: applyDatasetPayload,
  });
  const resetWatchlist = useMutation({
    mutationFn: () => api.saveWatchlist([], true, adminToken),
    onSuccess: (payload) => {
      setWatchlistText("");
      applyDatasetPayload(payload);
    },
  });
  const startRefresh = useMutation({
    mutationFn: () => api.startDatasetRefresh({ refresh_mode: refreshMode, remote_url: remoteUrl, files: uploadFiles ?? undefined }, adminToken),
    onSuccess: applyDatasetPayload,
  });

  useEffect(() => {
    const symbols = admin.data?.watchlist_symbols ?? [];
    if (!watchlistText && symbols.length) {
      setWatchlistText(symbols.join(", "));
    }
  }, [admin.data?.watchlist_symbols, watchlistText]);

  if (admin.isLoading) {
    return <LoadingBlock label="Loading data admin..." />;
  }
  if (admin.error) {
    return <ErrorBlock error={admin.error} />;
  }

  const payload = admin.data;
  const summary = payload?.summary ?? {};
  const status = payload?.status;
  const isUnlocked = Boolean(adminToken);
  const mutationError = unlock.error ?? saveWatchlist.error ?? resetWatchlist.error ?? startRefresh.error;
  const latestProblems = (payload?.latest ?? []).filter((row) => row.severity === "warn" || row.severity === "severe");
  const refreshJobs = payload?.refresh_jobs ?? [];
  const activeJob = refreshJobs.find((row) => row.job_id === status?.active_job_id) ?? refreshJobs[refreshJobs.length - 1];

  return (
    <section className="page-grid">
      <article className="panel panel--wide">
        <div className="panel-heading">
          <div>
            <h2>Data Admin Console</h2>
            <p>
              Manage stored datasets, watchlist symbols, CSV inputs, refresh jobs, and warn-only data health from the web UI.
              Refresh jobs reuse the same lock as the scheduler and Streamlit controls.
            </p>
          </div>
          <StatusPill label={status?.operating_mode ?? "Unknown mode"} tone={summary.overall_severity === "severe" ? "severe" : summary.overall_severity === "warn" ? "warn" : "ok"} />
        </div>
        <div className="filter-grid filter-grid--compact">
          <label>
            Admin unlock
            <input
              type="password"
              placeholder={payload?.admin.mode === "private" ? "Private mode: password optional" : "Admin password"}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>
          <label>
            Session status
            <button
              className="action-button"
              type="button"
              aria-label={isUnlocked ? "Refresh admin token" : "Unlock admin writes"}
              onClick={() => unlock.mutate()}
              disabled={unlock.isPending}
            >
              {isUnlocked ? "Refresh admin token" : "Unlock admin writes"}
            </button>
          </label>
          <label>
            Write state
            <span className="form-readout">{isUnlocked ? "Unlocked for this browser session" : "Read-only until unlocked"}</span>
          </label>
          <label>
            Scheduler
            <span className="form-readout">{status?.scheduler_enabled ? "Enabled" : "Disabled"}</span>
          </label>
        </div>
        {mutationError ? <ErrorBlock error={mutationError} /> : null}
      </article>

      <div className="metric-grid">
        <MetricCard label="Overall health" value={summary.overall_severity} />
        <MetricCard label="Missing core datasets" value={status?.missing_core_dataset_count ?? 0} />
        <MetricCard label="Last refresh" value={summary.last_refresh_status} caption={String(summary.last_refresh_mode ?? "")} />
        <MetricCard label="Active/latest job" value={activeJob?.status ?? "n/a"} caption={String(activeJob?.refresh_mode ?? "")} />
      </div>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Runtime status</h2>
          <StatusPill label={payload?.admin.mode ?? "private"} />
        </div>
        <dl className="detail-list">
          <div>
            <dt>State root</dt>
            <dd>{status?.state_root ?? "n/a"}</dd>
          </div>
          <div>
            <dt>Database</dt>
            <dd>{status?.db_path ?? "n/a"}</dd>
          </div>
          <div>
            <dt>Source mode</dt>
            <dd>
              {status?.source_mode.mode ?? "unknown"} ({status?.source_mode.truth_post_count ?? 0} Truth,{" "}
              {status?.source_mode.x_post_count ?? 0} X)
            </dd>
          </div>
        </dl>
        {status?.missing_core_datasets.length ? (
          <div className="empty-state">Missing core datasets: {status.missing_core_datasets.join(", ")}</div>
        ) : null}
      </article>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Watchlist controls</h2>
          <StatusPill label={`${payload?.watchlist_symbols.length ?? 0} symbols`} />
        </div>
        <div className="filter-grid filter-grid--compact">
          <label className="filter-grid__wide">
            Watchlist symbols
            <textarea
              value={watchlistText}
              onChange={(event) => setWatchlistText(event.target.value)}
              placeholder="NVDA, TSLA, XOM"
              disabled={!isUnlocked}
            />
          </label>
          <label>
            Save
            <button
              className="action-button"
              type="button"
              aria-label="Save watchlist"
              disabled={!isUnlocked || saveWatchlist.isPending}
              onClick={() => saveWatchlist.mutate()}
            >
              Save watchlist
            </button>
          </label>
          <label>
            Reset
            <button
              className="action-button action-button--danger"
              type="button"
              aria-label="Reset watchlist"
              disabled={!isUnlocked || resetWatchlist.isPending}
              onClick={() => resetWatchlist.mutate()}
            >
              Reset watchlist
            </button>
          </label>
        </div>
        <DataTable rows={payload?.asset_universe ?? []} emptyLabel="No asset universe rows yet." />
      </article>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Refresh controls</h2>
          <StatusPill label="Background job" />
        </div>
        <p>
          Refreshes run against stored state and are single-instance. A manual refresh here does not bypass the scheduler lock.
        </p>
        <div className="filter-grid filter-grid--compact">
          <label>
            Refresh mode
            <select value={refreshMode} onChange={(event) => setRefreshMode(event.target.value as "bootstrap" | "full" | "incremental")} disabled={!isUnlocked}>
              <option value="bootstrap">Bootstrap</option>
              <option value="full">Full</option>
              <option value="incremental">Incremental</option>
            </select>
          </label>
          <label className="filter-grid__wide">
            Remote X / mention CSV URL
            <input value={remoteUrl} onChange={(event) => setRemoteUrl(event.target.value)} placeholder="https://..." disabled={!isUnlocked} />
          </label>
          <label>
            CSV upload
            <input type="file" accept=".csv,text/csv" multiple disabled={!isUnlocked} onChange={(event) => setUploadFiles(event.target.files)} />
          </label>
          <label>
            Start
            <button
              className="action-button"
              type="button"
              aria-label="Start refresh job"
              disabled={!isUnlocked || startRefresh.isPending}
              onClick={() => startRefresh.mutate()}
            >
              Start refresh job
            </button>
          </label>
        </div>
      </article>

      <article className="panel panel--wide chart-grid">
        <div>
          <h2>Recent refresh jobs</h2>
          <DataTable rows={payload?.refresh_jobs ?? []} emptyLabel="No React-submitted refresh jobs yet." />
        </div>
        <div>
          <h2>Refresh history</h2>
          <DataTable rows={payload?.refresh_history ?? []} emptyLabel="No refresh history yet." />
        </div>
      </article>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Latest warn/severe diagnostics</h2>
          <StatusPill label={`${latestProblems.length} current issue(s)`} tone={latestProblems.some((row) => row.severity === "severe") ? "severe" : latestProblems.length ? "warn" : "ok"} />
        </div>
        <DataTable rows={latestProblems} emptyLabel="No warn or severe health rows." />
      </article>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Dataset registry</h2>
          <StatusPill label={`${payload?.registry.length ?? 0} datasets`} />
        </div>
        <DataTable rows={payload?.registry ?? []} />
      </article>

      <article className="panel panel--wide chart-grid">
        <div>
          <h2>Source manifest</h2>
          <DataTable rows={payload?.source_manifests ?? []} emptyLabel="No source manifest rows yet." />
        </div>
        <div>
          <h2>Asset market manifest</h2>
          <DataTable rows={payload?.asset_market_manifest ?? []} emptyLabel="No asset market manifest rows yet." />
        </div>
      </article>
    </section>
  );
}

function LiveDecisionPage() {
  const queryClient = useQueryClient();
  const live = useQuery({ queryKey: ["live-ops"], queryFn: api.liveOps, refetchInterval: 60_000 });
  const [adminToken, setAdminToken] = useState(() =>
    typeof window === "undefined" ? "" : window.sessionStorage.getItem("allcaps_admin_token") ?? "",
  );
  const [password, setPassword] = useState("");
  const [selectedRunId, setSelectedRunId] = useState("");
  const [fallbackMode, setFallbackMode] = useState<"SPY" | "FLAT">("SPY");
  const [startingCash, setStartingCash] = useState(100000);
  const [resetCash, setResetCash] = useState(100000);

  const applyLivePayload = (payload: LiveOpsPayload) => {
    queryClient.setQueryData(["live-ops"], payload);
  };
  const unlock = useMutation({
    mutationFn: () => api.adminSession(password),
    onSuccess: (payload) => {
      setAdminToken(payload.token);
      if (typeof window !== "undefined") {
        window.sessionStorage.setItem("allcaps_admin_token", payload.token);
      }
      setPassword("");
    },
  });
  const saveConfig = useMutation({
    mutationFn: () => api.saveLiveConfig({ portfolio_run_id: selectedRunId, fallback_mode: fallbackMode }, adminToken),
    onSuccess: applyLivePayload,
  });
  const capture = useMutation({
    mutationFn: () => api.captureLive(adminToken),
    onSuccess: applyLivePayload,
  });
  const paperAction = useMutation({
    mutationFn: (request: { action: "enable" | "disable" | "reset" | "archive"; starting_cash?: number }) =>
      api.paperCurrentAction(request, adminToken),
    onSuccess: applyLivePayload,
  });

  useEffect(() => {
    const config = live.data?.current_config ?? live.data?.seeded_config;
    const nextRunId = String(config?.portfolio_run_id ?? live.data?.run_options[0]?.run_id ?? "");
    const nextFallback = String(config?.fallback_mode ?? "SPY").toUpperCase() === "FLAT" ? "FLAT" : "SPY";
    if (nextRunId && !selectedRunId) {
      setSelectedRunId(nextRunId);
    }
    setFallbackMode(nextFallback);
    const activeStartingCash = Number(live.data?.paper.active_config?.starting_cash ?? 100000);
    if (Number.isFinite(activeStartingCash)) {
      setResetCash(activeStartingCash);
    }
  }, [live.data, selectedRunId]);

  if (live.isLoading) {
    return <LoadingBlock label="Loading live ops..." />;
  }
  if (live.error) {
    return <ErrorBlock error={live.error} />;
  }

  const payload = live.data;
  const decision = payload?.decision ?? {};
  const activePaper = payload?.paper.active_config ?? null;
  const isUnlocked = Boolean(adminToken);
  const mutationError = unlock.error ?? saveConfig.error ?? capture.error ?? paperAction.error;
  return (
    <section className="page-grid">
      <article className="panel panel--wide">
        <div className="panel-heading">
          <div>
            <h2>Live Ops Console</h2>
            <p>
              Configure the pinned joint portfolio run, capture stored-data live snapshots, and manage the matching
              paper portfolio. Dataset refreshes and model training remain in Streamlit for now.
            </p>
          </div>
          <StatusPill label={payload?.admin.capture_scope === "stored_data_only" ? "Stored-data capture only" : "Live ops"} />
        </div>
        <div className="filter-grid filter-grid--compact">
          <label>
            Admin unlock
            <input
              type="password"
              placeholder={payload?.admin.mode === "private" ? "Private mode: password optional" : "Admin password"}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>
          <label>
            Session status
            <button
              className="action-button"
              type="button"
              aria-label={isUnlocked ? "Refresh admin token" : "Unlock admin writes"}
              onClick={() => unlock.mutate()}
              disabled={unlock.isPending}
            >
              {isUnlocked ? "Refresh admin token" : "Unlock admin writes"}
            </button>
          </label>
          <label>
            Write state
            <span className="form-readout">{isUnlocked ? "Unlocked for this browser session" : "Read-only until unlocked"}</span>
          </label>
          <label>
            App mode
            <span className="form-readout">{payload?.admin.mode ?? "unknown"}</span>
          </label>
        </div>
        {mutationError ? <ErrorBlock error={mutationError} /> : null}
      </article>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Pinned portfolio run</h2>
          <StatusPill label={payload?.configured ? "Configured" : "Setup needed"} tone={payload?.configured ? "ok" : "warn"} />
        </div>
        {(payload?.errors ?? []).map((error) => (
          <p key={error}>{error}</p>
        ))}
        {payload?.run_options.length ? (
          <div className="filter-grid filter-grid--compact">
            <label>
              Pinned joint portfolio run
              <select value={selectedRunId} onChange={(event) => setSelectedRunId(event.target.value)} disabled={!isUnlocked}>
                {payload.run_options.map((run) => (
                  <option key={String(run.run_id)} value={String(run.run_id)}>
                    {String(run.run_name ?? run.run_id)} | {String(run.deployment_variant ?? "n/a")} |{" "}
                    {String(run.selected_symbols ?? "n/a")}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Fallback mode
              <select value={fallbackMode} onChange={(event) => setFallbackMode(event.target.value as "SPY" | "FLAT")} disabled={!isUnlocked}>
                <option value="SPY">SPY</option>
                <option value="FLAT">FLAT</option>
              </select>
            </label>
            <label>
              Config action
              <button
                className="action-button"
                type="button"
                aria-label="Save pinned portfolio run"
                disabled={!isUnlocked || !selectedRunId || saveConfig.isPending}
                onClick={() => saveConfig.mutate()}
              >
                Save pinned portfolio run
              </button>
            </label>
            <label>
              Capture action
              <button className="action-button" type="button" aria-label="Capture current board" disabled={!isUnlocked || capture.isPending} onClick={() => capture.mutate()}>
                Capture current board
              </button>
            </label>
          </div>
        ) : (
          <div className="empty-state">Save a joint portfolio model run before configuring Live Ops.</div>
        )}
        <DataTable rows={payload?.run_options ?? []} emptyLabel="No joint portfolio runs returned." />
      </article>

      <div className="metric-grid">
        <MetricCard label="Winner" value={decision.winning_asset ?? "FLAT"} />
        <MetricCard label="Decision source" value={decision.decision_source} />
        <MetricCard label="Winner score" value={decision.winner_score} />
        <MetricCard label="Eligible assets" value={decision.eligible_asset_count} />
      </div>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Current ranked board</h2>
          <StatusPill label={payload?.warnings.length ? `${payload.warnings.length} warning(s)` : "Current"} />
        </div>
        {(payload?.warnings ?? []).map((warning) => (
          <p key={warning}>{warning}</p>
        ))}
        <DataTable rows={payload?.board ?? []} />
      </article>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Capture status</h2>
          <StatusPill label="No ingestion refresh" />
        </div>
        <div className="metric-grid">
          <MetricCard label="Asset rows persisted" value={payload?.capture_result.persisted_assets ?? 0} />
          <MetricCard label="Decision rows persisted" value={payload?.capture_result.persisted_decisions ?? 0} />
          <MetricCard label="Paper decisions captured" value={payload?.capture_result.captured ?? 0} />
          <MetricCard label="Paper decisions settled" value={payload?.capture_result.settled ?? 0} />
        </div>
      </article>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Paper portfolio controls</h2>
          <StatusPill label={activePaper ? (activePaper.enabled ? "Enabled" : "Disabled") : "Not enabled"} tone={activePaper ? "ok" : "warn"} />
        </div>
        <div className="metric-grid">
          <MetricCard label="Paper portfolio" value={activePaper?.paper_portfolio_id ?? "none"} />
          <MetricCard label="Starting cash" value={activePaper?.starting_cash ?? startingCash} />
          <MetricCard label="Trade cost bps" value={activePaper?.transaction_cost_bps ?? "n/a"} />
          <MetricCard label="Archived portfolios" value={(payload?.paper.portfolios ?? []).filter((row) => row.archived_at).length} />
        </div>
        <div className="filter-grid filter-grid--compact">
          <label>
            Starting cash
            <input type="number" min={1000} step={1000} value={startingCash} onChange={(event) => setStartingCash(Number(event.target.value))} disabled={!isUnlocked} />
          </label>
          <label>
            Enable
            <button className="action-button" type="button" aria-label="Enable paper trading" disabled={!isUnlocked || paperAction.isPending} onClick={() => paperAction.mutate({ action: "enable", starting_cash: startingCash })}>
              Enable paper trading
            </button>
          </label>
          <label>
            Toggle
            <button
              className="action-button"
              type="button"
              aria-label={activePaper?.enabled ? "Disable paper trading" : "Enable existing portfolio"}
              disabled={!isUnlocked || !activePaper || paperAction.isPending}
              onClick={() => paperAction.mutate({ action: activePaper?.enabled ? "disable" : "enable", starting_cash: Number(activePaper?.starting_cash ?? startingCash) })}
            >
              {activePaper?.enabled ? "Disable paper trading" : "Enable existing portfolio"}
            </button>
          </label>
          <label>
            Archive
            <button className="action-button action-button--danger" type="button" aria-label="Archive current portfolio" disabled={!isUnlocked || !activePaper || paperAction.isPending} onClick={() => paperAction.mutate({ action: "archive" })}>
              Archive current portfolio
            </button>
          </label>
          <label>
            Reset cash
            <input type="number" min={1000} step={1000} value={resetCash} onChange={(event) => setResetCash(Number(event.target.value))} disabled={!isUnlocked || !activePaper} />
          </label>
          <label>
            Reset
            <button className="action-button" type="button" aria-label="Reset portfolio" disabled={!isUnlocked || paperAction.isPending} onClick={() => paperAction.mutate({ action: "reset", starting_cash: resetCash })}>
              Reset portfolio
            </button>
          </label>
        </div>
        <DataTable rows={payload?.paper.portfolios ?? []} emptyLabel="No paper portfolios have been created yet." />
      </article>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Recent live decisions</h2>
        </div>
        <DataTable rows={payload?.decision_history ?? []} emptyLabel="No persisted live decision history yet." />
      </article>

      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Recent paper journal</h2>
        </div>
        <DataTable rows={payload?.paper.decision_journal ?? []} emptyLabel="No paper decisions captured yet." />
      </article>
    </section>
  );
}

function PaperPerformancePage() {
  const portfolios = useQuery({ queryKey: ["paper-portfolios"], queryFn: api.paperPortfolios });
  const [selectedId, setSelectedId] = useState<string>("");
  const portfolioId = selectedId || String(portfolios.data?.current_config?.paper_portfolio_id ?? portfolios.data?.portfolios[0]?.paper_portfolio_id ?? "");
  const performance = useQuery({
    queryKey: ["performance", portfolioId],
    queryFn: () => api.performance(portfolioId),
    enabled: Boolean(portfolioId),
  });

  if (portfolios.isLoading) {
    return <LoadingBlock label="Loading paper portfolios..." />;
  }
  if (portfolios.error) {
    return <ErrorBlock error={portfolios.error} />;
  }
  if (!portfolios.data?.portfolios.length) {
    return <div className="empty-state">No paper portfolios have been created yet.</div>;
  }

  const summary = performance.data?.summary ?? {};
  return (
    <section className="page-grid">
      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Portfolio selector</h2>
          <StatusPill label={performance.data?.persisted ? "Persisted snapshot" : "In-memory snapshot"} />
        </div>
        <select value={portfolioId} onChange={(event) => setSelectedId(event.target.value)}>
          {portfolios.data.portfolios.map((portfolio) => (
            <option key={String(portfolio.paper_portfolio_id)} value={String(portfolio.paper_portfolio_id)}>
              {String(portfolio.paper_portfolio_id)} - {String(portfolio.portfolio_run_name ?? "portfolio")}
            </option>
          ))}
        </select>
      </article>
      {performance.isLoading ? <LoadingBlock label="Loading performance diagnostics..." /> : null}
      {performance.error ? <ErrorBlock error={performance.error} /> : null}
      {performance.data ? (
        <>
          <div className="metric-grid">
            <MetricCard label="Overall" value={summary.overall_severity} />
            <MetricCard label="Total return" value={summary.total_return} />
            <MetricCard label="Alpha" value={summary.alpha} />
            <MetricCard label="Fallback rate" value={summary.fallback_rate} />
          </div>
          <article className="panel panel--wide">
            <div className="panel-heading">
              <h2>Performance diagnostics</h2>
            </div>
            <DataTable rows={performance.data.diagnostics.filter((row) => row.severity !== "ok")} />
          </article>
          <article className="panel panel--wide">
            <div className="panel-heading">
              <h2>Winner distribution</h2>
            </div>
            <DataTable rows={performance.data.winner_distribution} />
          </article>
        </>
      ) : null}
    </section>
  );
}

export function App() {
  const [activePage, setActivePage] = useState<PageKey>("overview");
  const activeMeta = pages.find((page) => page.key === activePage) ?? pages[0];

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">AllCaps Web Migration</p>
          <h1>Web-first decision workbench</h1>
          <p>
            A React and FastAPI foundation that keeps the Python analytics engine intact
            while moving the user experience toward a production web application.
          </p>
        </div>
        <div className="hero-card">
          <span>Milestone</span>
          <strong>API foundation + React shell</strong>
          <small>Issue #43</small>
        </div>
      </header>

      <nav className="page-tabs" aria-label="Workbench sections">
        {pages.map((page) => (
          <button
            key={page.key}
            type="button"
            className={page.key === activePage ? "active" : ""}
            onClick={() => setActivePage(page.key)}
          >
            <span>{page.label}</span>
            <small>{page.deck}</small>
          </button>
        ))}
      </nav>

      <section className="section-heading">
        <p className="eyebrow">{activeMeta.label}</p>
        <h2>{activeMeta.deck}</h2>
      </section>

      {activePage === "overview" ? <OverviewPage /> : null}
      {activePage === "research" ? <ResearchPage /> : null}
      {activePage === "discovery" ? <DiscoveryPage /> : null}
      {activePage === "runs" ? <RunExplorerPage /> : null}
      {activePage === "data" ? <DataAdminPage /> : null}
      {activePage === "live" ? <LiveDecisionPage /> : null}
      {activePage === "paper" ? <PaperPerformancePage /> : null}
    </main>
  );
}
