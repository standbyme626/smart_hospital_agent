import { execFileSync } from "node:child_process";
import path from "node:path";

import {
  ADMIN_PASSWORD,
  ADMIN_USER,
  DATABASE_URL,
  PATIENT_PASSWORD,
  PATIENT_USER,
  PYTHON,
  REPO_ROOT,
} from "./env";

export default function globalSetup(): void {
  try {
    execFileSync(PYTHON, [path.join(REPO_ROOT, "scripts", "e2e_reset.py")], {
      stdio: "inherit",
      env: {
        ...process.env,
        E2E_DATABASE_URL: DATABASE_URL,
        ADMIN_USER,
        ADMIN_PASSWORD,
        PATIENT_USER,
        PATIENT_PASSWORD,
      },
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(
      `E2E database reset failed; check that Postgres is running and E2E_DATABASE_URL points at an isolated *_e2e database (${detail})`,
    );
  }
}
