import { defineConfig, devices } from "@playwright/test";

const testPort = process.env.ALLCAPS_FRONTEND_TEST_PORT ?? "57173";
const testBaseUrl = `http://127.0.0.1:${testPort}`;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  workers: 2,
  timeout: 30_000,
  expect: {
    timeout: 5_000,
  },
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: testBaseUrl,
    trace: "on-first-retry",
    serviceWorkers: "block",
  },
  webServer: {
    command: `npm run dev -- --port ${testPort}`,
    url: testBaseUrl,
    reuseExistingServer: false,
    timeout: 60_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
