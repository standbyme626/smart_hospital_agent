import { expect, test, type Page } from "@playwright/test";

import { ADMIN_PASSWORD, ADMIN_USER, PATIENT_PASSWORD, PATIENT_USER } from "./env";

test.describe.configure({ mode: "serial" });

async function open(page: Page, url: string): Promise<void> {
  await page.goto(url);
  await page.waitForLoadState("networkidle");
}

async function login(page: Page, username: string, password: string): Promise<void> {
  await open(page, "/login/");
  const home = page.getByText("哪里不舒服？");
  for (let attempt = 0; attempt < 3; attempt += 1) {
    await page.getByPlaceholder("请输入账号").fill(username);
    await page.getByPlaceholder("请输入密码").fill(password);
    const consent = page.getByRole("checkbox", { name: /我已阅读并同意/ });
    if (await consent.count()) {
      await consent.check();
    }
    await page.getByRole("button", { name: "登录", exact: true }).click();
    try {
      await home.waitFor({ state: "visible", timeout: 20_000 });
      return;
    } catch {
      await page.waitForLoadState("networkidle");
    }
  }
  await expect(home).toBeVisible();
}

test("follow-up creates a message and both pages render", async ({ page }) => {
  await login(page, PATIENT_USER, PATIENT_PASSWORD);

  await open(page, "/followups/");
  await expect(page.getByRole("heading", { name: "随访", exact: true })).toBeVisible();
  const title = "E2E 复诊提醒 " + Date.now();
  await page.getByPlaceholder("例如：2 周后复诊心内科").fill(title);
  await page.getByRole("button", { name: "添加", exact: true }).click();
  await expect(page.getByText(title)).toBeVisible({ timeout: 20_000 });

  await open(page, "/messages/");
  await expect(page.getByRole("heading", { name: "消息", exact: true })).toBeVisible();
  await expect(page.getByText("随访已安排").first()).toBeVisible({ timeout: 20_000 });
});

test("clinician and ops pages load for admin and are denied for patient", async ({ page }) => {
  await login(page, ADMIN_USER, ADMIN_PASSWORD);

  await open(page, "/clinician/");
  await expect(page.getByText("医生工作台 · 只读")).toBeVisible();
  await expect(page.getByText("今日患者")).toBeVisible();

  await open(page, "/ops/");
  await expect(page.getByText("运营 / 质控台")).toBeVisible();
  await expect(page.getByText("待审核")).toBeVisible();
  await expect(page.getByText("对账差异").first()).toBeVisible();

  await page.context().clearCookies();
  await page.evaluate(() => window.localStorage.clear());
  await login(page, PATIENT_USER, PATIENT_PASSWORD);

  await open(page, "/ops/");
  await expect(page.getByText("当前账号无运营权限")).toBeVisible();
});
