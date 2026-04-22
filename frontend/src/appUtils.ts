import type { ReplaySessionPayload, RecordRow } from "./api";

export function formatValue(value: unknown): string {
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

export function selectedOptions(options: HTMLCollectionOf<HTMLOptionElement>): string[] {
  return Array.from(options)
    .filter((option) => option.selected)
    .map((option) => option.value);
}

export function parseSymbolList(value: string): string[] {
  return value
    .split(/[\s,]+/)
    .map((item) => item.trim().toUpperCase())
    .filter(Boolean);
}

export function recordValue(value: unknown): string | number | boolean | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  return JSON.stringify(value);
}

export function deploymentParamRows(payload?: ReplaySessionPayload): RecordRow[] {
  const params = payload?.metadata?.deployment_params;
  if (!params || typeof params !== "object") {
    return [];
  }
  return Object.entries(params as Record<string, unknown>).map(([parameter, value]) => ({
    parameter,
    value: recordValue(value),
  }));
}

export function modelRunLabel(row: RecordRow): string {
  const runName = String(row.run_name ?? row.run_id ?? "");
  const asset = String(row.target_asset ?? "");
  const score = row.robust_score !== undefined && row.robust_score !== null ? ` | robust ${formatValue(row.robust_score)}` : "";
  return `${asset} | ${runName}${score}`;
}

export function modelWorkflowLabel(mode: "single_asset" | "saved_run_portfolio" | "joint_portfolio"): string {
  if (mode === "single_asset") {
    return "Single Asset";
  }
  if (mode === "saved_run_portfolio") {
    return "Saved-Run Portfolio";
  }
  return "Joint Portfolio";
}

export function parseModelJobSummary(job: RecordRow | undefined): Record<string, unknown> {
  const summary = job?.summary;
  if (!summary) {
    return {};
  }
  if (typeof summary === "object") {
    return summary as Record<string, unknown>;
  }
  try {
    return JSON.parse(String(summary)) as Record<string, unknown>;
  } catch {
    return {};
  }
}
