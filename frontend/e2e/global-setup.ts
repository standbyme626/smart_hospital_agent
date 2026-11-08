import { execFileSync } from "node:child_process";
import path from "node:path";

import { ADMIN_PASSWORD, DATABASE_URL, PATIENT_PASSWORD, PYTHON, REPO_ROOT } from "./env";

export default function globalSetup(): void {
  execFileSync(PYTHON, [path.join(REPO_ROOT, "scripts", "e2e_reset.py")], {
    stdio: "inherit",
    env: {
      ...process.env,
      E2E_DATABASE_URL: DATABASE_URL,
      ADMIN_PASSWORD,
      PATIENT_PASSWORD,
    },
  });
}
