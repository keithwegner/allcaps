import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type RecordRow } from "./api";

type PageKey = "overview" | "data" | "live" | "paper";

const pages: Array<{ key: PageKey; label: string; deck: string }> = [
  { key: "overview", label: "Overview", deck: "API status and migration posture" },
  { key: "data", label: "Data Health", deck: "Freshness, completeness, and anomalies" },
  { key: "live", label: "Live Decision", deck: "Current portfolio board" },
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

function DataHealthPage() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  if (health.isLoading) {
    return <LoadingBlock label="Loading data health..." />;
  }
  if (health.error) {
    return <ErrorBlock error={health.error} />;
  }

  const summary = health.data?.summary ?? {};
  return (
    <section className="page-grid">
      <div className="metric-grid">
        <MetricCard label="Overall" value={summary.overall_severity} />
        <MetricCard label="Severe checks" value={summary.severe_count} />
        <MetricCard label="Warn checks" value={summary.warn_count} />
        <MetricCard label="Last refresh" value={summary.last_refresh_status} />
      </div>
      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Latest warn/severe diagnostics</h2>
          <StatusPill label={`${health.data?.latest.length ?? 0} checks`} />
        </div>
        <DataTable
          rows={(health.data?.latest ?? []).filter((row) => row.severity === "warn" || row.severity === "severe")}
          emptyLabel="No warn or severe health rows."
        />
      </article>
      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Dataset registry</h2>
        </div>
        <DataTable rows={health.data?.registry ?? []} />
      </article>
    </section>
  );
}

function LiveDecisionPage() {
  const live = useQuery({ queryKey: ["live"], queryFn: api.live, refetchInterval: 60_000 });
  if (live.isLoading) {
    return <LoadingBlock label="Loading live decision..." />;
  }
  if (live.error) {
    return <ErrorBlock error={live.error} />;
  }

  if (!live.data?.configured) {
    return (
      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Live monitor is not configured</h2>
          <StatusPill label="Setup needed" tone="warn" />
        </div>
        {(live.data?.errors ?? []).map((error) => (
          <p key={error}>{error}</p>
        ))}
      </article>
    );
  }

  const decision = live.data.decision ?? {};
  return (
    <section className="page-grid">
      <div className="metric-grid">
        <MetricCard label="Winner" value={decision.winning_asset ?? "FLAT"} />
        <MetricCard label="Decision source" value={decision.decision_source} />
        <MetricCard label="Winner score" value={decision.winner_score} />
        <MetricCard label="Eligible assets" value={decision.eligible_asset_count} />
      </div>
      <article className="panel panel--wide">
        <div className="panel-heading">
          <h2>Current ranked board</h2>
          <StatusPill label="Read-only" />
        </div>
        <DataTable rows={live.data.board} />
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
      {activePage === "data" ? <DataHealthPage /> : null}
      {activePage === "live" ? <LiveDecisionPage /> : null}
      {activePage === "paper" ? <PaperPerformancePage /> : null}
    </main>
  );
}
