import { describe, expect, test } from "vitest";
import {
  deploymentParamRows,
  formatValue,
  modelRunLabel,
  modelWorkflowLabel,
  parseModelJobSummary,
  parseSymbolList,
  recordValue,
  selectedOptions,
} from "../src/appUtils";

describe("app utility helpers", () => {
  test("formats display values consistently", () => {
    expect(formatValue(null)).toBe("n/a");
    expect(formatValue(undefined)).toBe("n/a");
    expect(formatValue("")).toBe("n/a");
    expect(formatValue(0.1234)).toBe("12.34%");
    expect(formatValue(12)).toBe("12");
    expect(formatValue(12.34567)).toBe("12.3457");
    expect(formatValue("SPY")).toBe("SPY");
  });

  test("parses symbols and selected DOM options", () => {
    expect(parseSymbolList("spy, qqq\nnvda  tsla")).toEqual(["SPY", "QQQ", "NVDA", "TSLA"]);

    const select = document.createElement("select");
    select.multiple = true;
    ["SPY", "QQQ", "NVDA"].forEach((value, index) => {
      const option = document.createElement("option");
      option.value = value;
      option.selected = index !== 1;
      select.append(option);
    });

    expect(selectedOptions(select.selectedOptions)).toEqual(["SPY", "NVDA"]);
  });

  test("normalizes replay deployment parameter rows", () => {
    expect(recordValue(null)).toBeNull();
    expect(recordValue(true)).toBe(true);
    expect(recordValue({ fallback: "SPY" })).toBe('{"fallback":"SPY"}');
    expect(deploymentParamRows()).toEqual([]);
    expect(
      deploymentParamRows({
        ready: true,
        message: "",
        run_id: "run-1",
        signal_session_date: "2025-01-03",
        metadata: {
          deployment_params: {
            threshold: 0.01,
            enabled: true,
            nested: { min_post_count: 2 },
          },
        },
        metrics: {},
        prediction: [],
        comparison_rows: [],
        feature_importance: [],
        feature_contributions: [],
        post_attribution: [],
        account_attribution: [],
      }),
    ).toEqual([
      { parameter: "threshold", value: 0.01 },
      { parameter: "enabled", value: true },
      { parameter: "nested", value: '{"min_post_count":2}' },
    ]);
  });

  test("labels model rows and workflow modes", () => {
    expect(modelRunLabel({ run_id: "run-1", run_name: "Baseline", target_asset: "SPY", robust_score: 0.1234 })).toBe(
      "SPY | Baseline | robust 12.34%",
    );
    expect(modelRunLabel({ run_id: "run-2" })).toBe(" | run-2");
    expect(modelWorkflowLabel("single_asset")).toBe("Single Asset");
    expect(modelWorkflowLabel("saved_run_portfolio")).toBe("Saved-Run Portfolio");
    expect(modelWorkflowLabel("joint_portfolio")).toBe("Joint Portfolio");
  });

  test("parses model job summaries safely", () => {
    expect(parseModelJobSummary(undefined)).toEqual({});
    expect(parseModelJobSummary({ summary: { metrics: { total_return: 0.1 } } })).toEqual({ metrics: { total_return: 0.1 } });
    expect(parseModelJobSummary({ summary: '{"run_id":"run-1"}' })).toEqual({ run_id: "run-1" });
    expect(parseModelJobSummary({ summary: "{bad json" })).toEqual({});
  });
});
