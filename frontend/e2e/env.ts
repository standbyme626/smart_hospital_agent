import path from "node:path";

export const FRONTEND_DIR = path.resolve(__dirname, "..");
export const REPO_ROOT = path.resolve(FRONTEND_DIR, "..");

function resolvePython(value: string): string {
  return value.includes("/") ? path.resolve(FRONTEND_DIR, value) : value;
}

export const PYTHON = resolvePython(process.env.E2E_PYTHON || "../backend/.venv/bin/python");
export const BACKEND_PORT = Number(process.env.E2E_BACKEND_PORT || 8020);
export const FRONTEND_PORT = Number(process.env.E2E_FRONTEND_PORT || 3000);
export const MODEL_PORT = Number(process.env.E2E_MODEL_PORT || 9999);
export const MCP_PORT = Number(process.env.E2E_MCP_PORT || 9101);

export const DATABASE_URL =
  process.env.E2E_DATABASE_URL || "postgresql://sha:sha_pwd@127.0.0.1:5434/sha_e2e";

function assertE2eDatabase(url: string): void {
  let database = "";
  try {
    database = new URL(url).pathname.replace(/^\//, "");
  } catch {
    throw new Error("E2E_DATABASE_URL is not a valid database URL; expected postgresql://user:pass@host:port/sha_e2e");
  }
  if (!database.endsWith("_e2e")) {
    throw new Error(
      `Refusing to run E2E against database "${database}": the E2E database name must end with "_e2e" ` +
        `(production "sha" and test "sha_test" are forbidden). Set E2E_DATABASE_URL to an isolated *_e2e database.`,
    );
  }
}

assertE2eDatabase(DATABASE_URL);

export const ADMIN_USER = "admin";
export const ADMIN_PASSWORD = process.env.E2E_ADMIN_PASSWORD || "e2e-admin-pass";
export const PATIENT_USER = "patient";
export const PATIENT_PASSWORD = process.env.E2E_PATIENT_PASSWORD || "e2e-patient-pass";

export const API_BASE = `http://127.0.0.1:${BACKEND_PORT}`;
export const FRONTEND_URL = `http://127.0.0.1:${FRONTEND_PORT}`;
