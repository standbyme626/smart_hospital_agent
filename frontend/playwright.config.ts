import path from "node:path";

import { defineConfig } from "@playwright/test";

import {
  ADAPTER_PORT,
  ADMIN_PASSWORD,
  API_BASE,
  BACKEND_PORT,
  DATABASE_URL,
  FRONTEND_DIR,
  FRONTEND_PORT,
  FRONTEND_URL,
  MODEL_PORT,
  PATIENT_PASSWORD,
  PYTHON,
  REPO_ROOT,
} from "./e2e/env";

const BACKEND_DIR = path.join(REPO_ROOT, "backend");
const HITL_CONFIRM_TOKEN = "e2e-confirm";
const MODEL_BASE = `http://127.0.0.1:${MODEL_PORT}/v1`;

export default defineConfig({
  testDir: "./e2e",
  timeout: 180_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  globalSetup: "./e2e/global-setup.ts",
  use: {
    baseURL: FRONTEND_URL,
    locale: "zh-CN",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: [
    {
      command: `"${PYTHON}" "${path.join(REPO_ROOT, "scripts", "fake_model_server.py")}" --port ${MODEL_PORT}`,
      url: `http://127.0.0.1:${MODEL_PORT}/health`,
      reuseExistingServer: false,
      timeout: 30_000,
    },
    {
      command: `"${PYTHON}" -m his_adapter`,
      cwd: BACKEND_DIR,
      url: `http://127.0.0.1:${ADAPTER_PORT}/health`,
      env: {
        HIS_ADAPTER_MODE: "mock",
        HIS_PAYMENT_MODE: "cashier",
        ADAPTER_DATABASE_URL: DATABASE_URL,
        ADAPTER_HOST: "127.0.0.1",
        ADAPTER_PORT: String(ADAPTER_PORT),
      },
      reuseExistingServer: false,
      timeout: 60_000,
    },
    {
      command: `"${PYTHON}" -m uvicorn app.main:app --host 127.0.0.1 --port ${BACKEND_PORT}`,
      cwd: BACKEND_DIR,
      url: `${API_BASE}/api/health`,
      env: {
        DATABASE_URL,
        AUTH_ENABLED: "true",
        // E2E 要验证"诊毕/签名/处方"完整链路，HIS 侧是本地 mock，安全
        CLINICIAN_WRITE_ENABLED: "true",
        JWT_SECRET: "e2e-jwt-secret",
        ADMIN_USER: "admin",
        ADMIN_PASSWORD,
        PATIENT_USER: "patient",
        PATIENT_PASSWORD,
        OPENCODE_API_BASE: MODEL_BASE,
        OPENCODE_API_KEY: "fake-key",
        SILICONFLOW_API_BASE: MODEL_BASE,
        SILICONFLOW_API_KEY: "fake-key",
        OLLAMA_ENABLED: "false",
        HOSPITAL_MCP_URL: `http://127.0.0.1:${ADAPTER_PORT}/mcp`,
        HITL_CONFIRM_TOKEN,
        HIS_PAYMENT_MODE: "mock",
        RATE_LIMIT_PER_MINUTE: "10000",
        CHAT_DAILY_LIMIT: "1000",
        CORS_ORIGINS: `http://localhost:${FRONTEND_PORT},http://127.0.0.1:${FRONTEND_PORT}`,
        UPLOADS_DIR: "/tmp/sha-e2e-uploads",
      },
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: `npm run dev -- -p ${FRONTEND_PORT}`,
      cwd: FRONTEND_DIR,
      url: FRONTEND_URL,
      env: { NEXT_PUBLIC_API_BASE: API_BASE, NEXT_TELEMETRY_DISABLED: "1", NEXT_DIST_DIR: process.env.NEXT_DIST_DIR || ".next" },
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
});
