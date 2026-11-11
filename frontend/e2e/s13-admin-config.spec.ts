import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

import { ADMIN_PASSWORD, ADMIN_USER, API_BASE, PATIENT_PASSWORD, PATIENT_USER } from "./env";

test.describe.configure({ mode: "serial" });

/** 直接注入 token，不依赖登录页（登录页在并行开发中，E2E 不应被它拖住）。 */
async function signIn(
  page: Page,
  request: APIRequestContext,
  username: string,
  password: string,
): Promise<void> {
  const response = await request.post(API_BASE + "/api/v1/auth/login", {
    data: { username, password },
  });
  expect(response.ok(), "登录接口失败：" + response.status()).toBeTruthy();
  const payload = (await response.json()) as { access_token: string };
  await page.addInitScript((token: string) => {
    window.localStorage.setItem("sha_token", token);
  }, payload.access_token);
}

async function open(page: Page, url: string): Promise<void> {
  await page.goto(url);
  await page.waitForLoadState("networkidle");
}

test("配置管理：患者被拒绝，管理员可进入", async ({ page }) => {
  await signIn(page, page.request, PATIENT_USER, PATIENT_PASSWORD);
  await open(page, "/platform/config/");
  await expect(page.getByText("需要管理员权限")).toBeVisible();
});

test("配置管理：管理员看到四种配置", async ({ page }) => {
  await signIn(page, page.request, ADMIN_USER, ADMIN_PASSWORD);
  await open(page, "/platform/config/");
  await expect(page.getByRole("heading", { name: "配置管理" })).toBeVisible();
  for (const name of [/规则库/, /通知模板/, /Prompt 管理/, /Agent 配置/]) {
    await expect(page.getByRole("button", { name })).toBeVisible();
  }
});

test("配置管理：新建规则 → 保存草稿 → 发布 → 版本历史", async ({ page }) => {
  await signIn(page, page.request, ADMIN_USER, ADMIN_PASSWORD);
  await open(page, "/platform/config/");

  const key = "e2e_rule_" + Date.now().toString(36);
  await page.getByLabel("新建条目 key").fill(key);
  await page.getByRole("button", { name: "新建" }).click();
  await expect(page.getByText(/已填入模板/)).toBeVisible();

  await page
    .getByLabel("草稿（JSON）")
    .fill(JSON.stringify({ severity: "high", keywords: ["e2e 关键词"], action: "e2e 动作" }, null, 2));

  await page.getByRole("button", { name: "保存草稿" }).click();
  await expect(page.getByText(/草稿已保存/)).toBeVisible();

  await page.getByPlaceholder("变更说明（写入审计）").fill("e2e 验证");
  await page.getByRole("button", { name: "发布" }).click();
  await expect(page.getByText(/已发布 v/)).toBeVisible();
  await expect(page.getByText("当前生效值")).toBeVisible();
  await expect(page.getByRole("button", { name: /回滚到此版/ })).toBeVisible();
});

test("配置管理：通知模板变量预览", async ({ page }) => {
  await signIn(page, page.request, ADMIN_USER, ADMIN_PASSWORD);
  await open(page, "/platform/config/");

  await page.getByRole("button", { name: /通知模板/ }).click();
  await page.getByRole("button", { name: /report_ready/ }).click();

  await page
    .getByLabel("模板预览变量（JSON）")
    .fill(JSON.stringify({ member: "张三", kind: "血常规", due: "11-20" }, null, 2));
  await page.getByRole("button", { name: "预览渲染结果" }).click();
  await expect(page.getByText(/张三的血常规报告已解读完成/)).toBeVisible();
});
