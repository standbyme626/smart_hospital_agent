import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

import { ADMIN_PASSWORD, ADMIN_USER, API_BASE, PATIENT_PASSWORD, PATIENT_USER } from "./env";

test.describe.configure({ mode: "serial" });

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

test("管理端权限：患者访问知识库被拒绝", async ({ page }) => {
  await signIn(page, page.request, PATIENT_USER, PATIENT_PASSWORD);
  await open(page, "/platform/knowledge/");
  await expect(page.getByText("需要管理员权限")).toBeVisible();
});

test("知识库：覆盖情况与检索调试可用", async ({ page }) => {
  await signIn(page, page.request, ADMIN_USER, ADMIN_PASSWORD);
  await open(page, "/platform/knowledge/");

  await expect(page.getByRole("heading", { name: "知识库与检索" })).toBeVisible();
  await expect(page.getByText("覆盖情况")).toBeVisible();
  await expect(page.getByText("文档总数")).toBeVisible();
  for (const name of [/知识库策略/, /语料与索引/, /未命中问题/, /检索调试/, /RAG 全链路/]) {
    await expect(page.getByRole("button", { name })).toBeVisible();
  }

  await page.getByRole("button", { name: /检索调试/ }).click();
  await page.getByRole("button", { name: "检索", exact: true }).click();
  await expect(page.getByText(/命中 \d+ 条 ·/)).toBeVisible({ timeout: 30_000 });
});

test("可观测：验收自测显示 25 条剧本看板", async ({ page }) => {
  await signIn(page, page.request, ADMIN_USER, ADMIN_PASSWORD);
  await open(page, "/platform/insight/");

  await expect(page.getByRole("heading", { name: "可观测与验收" })).toBeVisible();
  await page.getByRole("button", { name: /验收自测/ }).click();
  await expect(page.getByText("系统自检")).toBeVisible();
  await expect(page.getByText("25 条验收剧本")).toBeVisible();
  await expect(page.getByText(/可跑 10 · 部分 10 · 阻塞 5/)).toBeVisible();
  await expect(page.getByText("INT-06")).toBeVisible();
});

test("接入与实验：HIS 测试连接留痕，实验列表可进入", async ({ page }) => {
  await signIn(page, page.request, ADMIN_USER, ADMIN_PASSWORD);
  await open(page, "/platform/integrations/");

  await expect(page.getByRole("heading", { name: "接入与实验" })).toBeVisible();
  await page.getByRole("button", { name: "测试连接" }).click();
  await expect(page.getByText(/连接(正常|异常)：/)).toBeVisible({ timeout: 30_000 });

  await page.getByRole("button", { name: /实验与灰度/ }).click();
  await expect(page.getByText("实验列表")).toBeVisible();
  await expect(page.getByRole("button", { name: "创建" })).toBeVisible();
});

test("规范与数据流：状态机取值可读", async ({ page }) => {
  await signIn(page, page.request, ADMIN_USER, ADMIN_PASSWORD);
  await open(page, "/platform/spec/");

  await expect(page.getByRole("heading", { name: "规范与数据流" })).toBeVisible();
  await expect(page.getByText("数据流（实时计数）")).toBeVisible();

  await page.getByRole("button", { name: /状态规范/ }).click();
  await expect(page.getByText("状态机实际取值")).toBeVisible();
  await expect(page.getByText(/bookings\.status/)).toBeVisible();
});
