import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { api } from "../src/api";

type FetchCall = {
  url: string;
  init?: RequestInit;
};

const calls: FetchCall[] = [];

function mockFetch(status = 200, body: unknown = { ok: true }, statusText = "OK") {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    return new Response(JSON.stringify(body), {
      status,
      statusText,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("frontend API client", () => {
  beforeEach(() => {
    calls.length = 0;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test("builds GET URLs with encoded research filters", async () => {
    mockFetch(200, { ready: true });

    await api.research({
      platforms: ["Truth Social", "X"],
      keyword: "tariff rally",
      include_reshares: false,
      tracked_only: true,
      narrative_platforms: ["Truth Social"],
      narrative_topic: "",
    });

    const url = new URL(calls[0].url);
    expect(url.pathname).toBe("/api/research");
    expect(url.searchParams.getAll("platforms")).toEqual(["Truth Social", "X"]);
    expect(url.searchParams.get("keyword")).toBe("tariff rally");
    expect(url.searchParams.get("include_reshares")).toBe("false");
    expect(url.searchParams.get("tracked_only")).toBe("true");
    expect(url.searchParams.getAll("narrative_platforms")).toEqual(["Truth Social"]);
    expect(url.searchParams.has("narrative_topic")).toBe(false);
  });

  test("builds read endpoint URLs with encoded path and query values", async () => {
    mockFetch(200, { found: true });

    await api.runDetail("run/with slash", { variantName: "per asset", sessionDate: "2025-01-03" });
    await api.runComparison(["run-1", "run 2"], "run-1");
    await api.replay("run 1");
    await api.replaySession("run 1", "2025-01-03");
    await api.performance("paper 1");

    expect(calls[0].url).toContain("/api/runs/run%2Fwith%20slash");
    expect(calls[0].url).toContain("variant_name=per+asset");
    expect(calls[1].url).toContain("/api/runs/compare");
    expect(calls[1].url).toContain("run_ids=run-1");
    expect(calls[1].url).toContain("run_ids=run+2");
    expect(calls[2].url).toContain("/api/replay?run_id=run+1");
    expect(calls[3].url).toContain("/api/replay/session?run_id=run+1&signal_session_date=2025-01-03");
    expect(calls[4].url).toContain("/api/performance/paper%201");
  });

  test("sends JSON writes with bearer tokens", async () => {
    mockFetch(200, { token: "next" });

    await api.adminSession("secret");
    await api.saveWatchlist(["SPY", "QQQ"], false, "token-1");
    await api.startModelTrainingJob({ workflow_mode: "joint_portfolio", selected_symbols: ["SPY"] }, "token-2");
    await api.createDiscoveryOverride({ account_id: "acct-1", action: "pin", effective_from: "2025-01-01" }, "token-3");
    await api.deleteDiscoveryOverride("override 1", "token-4");
    await api.saveLiveConfig({ portfolio_run_id: "run-1", fallback_mode: "SPY" }, "token-5");
    await api.captureLive("token-6");
    await api.paperCurrentAction({ action: "reset", starting_cash: 100000 }, "token-7");

    expect(JSON.parse(String(calls[0].init?.body))).toEqual({ password: "secret" });
    expect(calls[1].init?.headers).toMatchObject({ Authorization: "Bearer token-1" });
    expect(JSON.parse(String(calls[1].init?.body))).toEqual({ symbols: ["SPY", "QQQ"], reset: false });
    expect(calls[2].init?.headers).toMatchObject({ Authorization: "Bearer token-2" });
    expect(calls[3].init?.headers).toMatchObject({ Authorization: "Bearer token-3" });
    expect(calls[4].url).toContain("/api/discovery/overrides/override%201");
    expect(calls[4].init?.method).toBe("DELETE");
    expect(calls[5].init?.headers).toMatchObject({ Authorization: "Bearer token-5" });
    expect(calls[6].init?.headers).toMatchObject({ Authorization: "Bearer token-6" });
    expect(JSON.parse(String(calls[7].init?.body))).toEqual({ action: "reset", starting_cash: 100000 });
  });

  test("sends dataset refresh uploads as form data", async () => {
    mockFetch(200, { job_id: "job-1" });
    const file = new File(["author,text\nmacro,hello\n"], "mentions.csv", { type: "text/csv" });

    await api.startDatasetRefresh({ refresh_mode: "full", remote_url: "https://example.invalid/data.csv", files: [file] }, "token");

    expect(calls[0].init?.method).toBe("POST");
    expect(calls[0].init?.headers).toMatchObject({ Authorization: "Bearer token" });
    const body = calls[0].init?.body;
    expect(body).toBeInstanceOf(FormData);
    expect((body as FormData).get("refresh_mode")).toBe("full");
    expect((body as FormData).get("remote_url")).toBe("https://example.invalid/data.csv");
    expect((body as FormData).getAll("files")).toHaveLength(1);
  });

  test("surfaces JSON and non-JSON HTTP errors", async () => {
    mockFetch(400, { detail: ["bad token", "missing field"] }, "Bad Request");
    await expect(api.saveWatchlist([], false, "bad")).rejects.toThrow("bad token; missing field");

    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("not-json", { status: 502, statusText: "Bad Gateway" })),
    );
    await expect(api.status()).rejects.toThrow("502 Bad Gateway");
  });

  test("surfaces delete and upload HTTP errors", async () => {
    mockFetch(409, { detail: "override locked" }, "Conflict");
    await expect(api.deleteDiscoveryOverride("override-1", "token")).rejects.toThrow("override locked");

    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("not-json", { status: 503, statusText: "Unavailable" })),
    );
    await expect(api.deleteDiscoveryOverride("override-1", "token")).rejects.toThrow("503 Unavailable");

    mockFetch(422, { detail: ["bad file", "bad mode"] }, "Unprocessable Entity");
    await expect(api.startDatasetRefresh({ refresh_mode: "full" }, "token")).rejects.toThrow("bad file; bad mode");

    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("not-json", { status: 504, statusText: "Gateway Timeout" })),
    );
    await expect(api.startDatasetRefresh({ refresh_mode: "full" }, "token")).rejects.toThrow("504 Gateway Timeout");
  });

  test("exposes common read helpers and export URL", async () => {
    mockFetch(200, { ok: true });

    await api.status();
    await api.health();
    await api.datasetAdmin();
    await api.datasetRefreshJob("job 1");
    await api.modelTraining();
    await api.modelTrainingJob("model job");
    await api.runs();
    await api.researchAssets({ selected_asset: "NVDA", before_minutes: 120 });
    await api.discovery();
    await api.live();
    await api.liveOps();
    await api.paperPortfolios();

    expect(calls.map((call) => new URL(call.url).pathname)).toEqual([
      "/api/status",
      "/api/datasets/health",
      "/api/datasets/admin",
      "/api/datasets/jobs/job%201",
      "/api/models/training",
      "/api/models/jobs/model%20job",
      "/api/runs",
      "/api/research/assets",
      "/api/discovery",
      "/api/live/current",
      "/api/live/ops",
      "/api/paper/portfolios",
    ]);
    expect(api.researchExportUrl({ platforms: ["Truth Social"], keyword: "jobs" })).toContain(
      "/api/research/export?platforms=Truth+Social&keyword=jobs",
    );
  });
});
