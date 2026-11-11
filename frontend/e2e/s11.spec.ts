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

test("platform console loads for admin and is denied for patient", async ({ page }) => {
  await login(page, ADMIN_USER, ADMIN_PASSWORD);

  await open(page, "/platform/");
  await expect(page.getByText("平台管理").first()).toBeVisible();
  await expect(page.getByText("数据规模")).toBeVisible();
  await expect(page.getByText("知识文档")).toBeVisible();

  await page.getByRole("button", { name: "开关与配置" }).click();
  await expect(page.getByText("医生签名")).toBeVisible();
  await expect(page.getByText("未启用（仅草稿）").or(page.getByText("已启用"))).toBeVisible();

  await page.getByRole("button", { name: "能力矩阵" }).click();
  await expect(page.getByText("医保结算 / 电子处方")).toBeVisible();
  await expect(page.getByText("待确认", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "合规与数据" }).click();
  await expect(page.getByRole("heading", { name: "留存策略" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "脱敏规则" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "数据接口" })).toBeVisible();

  await page.getByRole("button", { name: "告警与 SLO" }).click();
  await expect(page.getByText("数据库与 Schema")).toBeVisible();
  await expect(page.getByText("医生写入门禁")).toBeVisible();

  await page.getByRole("button", { name: "质量评测" }).click();
  await expect(page.getByRole("heading", { name: "质量评测" })).toBeVisible();

  await page.evaluate(() => window.localStorage.clear());
  await login(page, PATIENT_USER, PATIENT_PASSWORD);

  await open(page, "/platform/");
  await expect(page.getByText("当前账号无管理权限")).toBeVisible();
});
