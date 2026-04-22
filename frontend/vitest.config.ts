import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: "./tests/setup.ts",
    exclude: ["e2e/**", "node_modules/**", "dist/**"],
    globals: false,
    coverage: {
      provider: "v8",
      reporter: ["text", "lcov", "html"],
      reportsDirectory: "./coverage",
      include: ["src/**/*.{ts,tsx}"],
      exclude: ["src/main.tsx", "src/vite-env.d.ts"],
        thresholds: {
          statements: 90,
          branches: 72,
          functions: 90,
          lines: 90,
        },
    },
  },
});
