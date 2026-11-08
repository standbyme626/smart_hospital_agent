import path from "node:path";

import { expect, test, type Page } from "@playwright/test";

import { ADMIN_PASSWORD, ADMIN_USER, PATIENT_PASSWORD, PATIENT_USER } from "./env";

const REPORT_MARKER = "【E2E】";
const MEMBER_NAME = "测试家人";
const PNG_PATH = path.resolve(__dirname, "fixtures/report.png");

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

async function switchMember(page: Page, currentName: string, nextName: string): Promise<void> {
  const header = page.locator("header");
  await header.getByRole("button", { name: new RegExp(`^${currentName}`) }).first().click();
  await header.getByRole("button", { name: new RegExp(`^${nextName}`) }).last().click();
}

test("consult wizard, follow-up question and session restore", async ({ page }) => {
  await login(page, PATIENT_USER, PATIENT_PASSWORD);
  await open(page, "/consult/");
  await expect(page.getByText("这次为谁咨询？")).toBeVisible();

  await page.getByRole("button", { name: "下一步" }).click();
  await expect(page.getByText("主要是哪个部位？")).toBeVisible();
  await page.getByRole("button", { name: "头面部" }).click();
  await page.getByRole("button", { name: "疼痛" }).click();
  await page.getByRole("button", { name: "下一步" }).click();
  await page.getByRole("button", { name: /1[–-]3 天/ }).click();
  await page.getByRole("button", { name: "3", exact: true }).click();
  await page.getByRole("button", { name: "下一步" }).click();
  await page.getByRole("button", { name: "没有这些情况" }).click();
  await page.getByRole("button", { name: "补充病史" }).click();
  await expect(page.getByText("补充关键信息")).toBeVisible();
  await page.getByRole("button", { name: "开始分诊" }).click();

  await expect(page.getByText("分诊结果")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText(`${REPORT_MARKER}结合你描述的症状`).first()).toBeVisible();
  await expect(page.getByText("心血管内科").first()).toBeVisible();

  const followup = page.locator("textarea").last();
  await followup.fill("需要做哪些检查？");
  await page.getByRole("button", { name: "发送追问" }).click();
  await expect(page.getByText("需要做哪些检查？", { exact: true })).toBeVisible();
  await expect(page.getByText(/会话已保存/)).toBeVisible({ timeout: 60_000 });

  const sessionUrl = page.url();
  expect(sessionUrl).toContain("?session=");
  await page.reload();
  await page.waitForLoadState("networkidle");
  await expect(page.getByText("分诊结果")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText(`${REPORT_MARKER}结合你描述的症状`).first()).toBeVisible();
});

test("report upload with fake vision and family member scoping", async ({ page }) => {
  await login(page, PATIENT_USER, PATIENT_PASSWORD);
  await open(page, "/reports/");
  await page.locator('input[type="file"]').setInputFiles(PNG_PATH);
  await expect(page.getByAltText("预览")).toBeVisible();
  await page.getByRole("button", { name: "开始解读" }).click();
  await expect(page.getByText(`${REPORT_MARKER}检查报告解读`).first()).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("异常指标")).toBeVisible();
  await expect(page.getByText("白细胞计数")).toBeVisible();
  const reportImage = page.getByAltText("报告原图");
  await expect(reportImage).toBeVisible();
  await expect
    .poll(() => reportImage.evaluate((img) => (img as HTMLImageElement).naturalWidth))
    .toBeGreaterThan(0);

  await open(page, "/profile/");
  await page.getByRole("button", { name: "添加", exact: true }).click();
  await page.getByPlaceholder("例如：张三").fill(MEMBER_NAME);
  await page.getByLabel("关系").selectOption("子女");
  await page.getByRole("button", { name: "保存" }).click();
  await expect(page.getByText(MEMBER_NAME).first()).toBeVisible();

  await switchMember(page, "本人", MEMBER_NAME);
  await open(page, "/reports/");
  await expect(page.getByText("还没有报告记录，上传一份试试。")).toBeVisible();
  await expect(page.getByText("0 条")).toBeVisible();

  await switchMember(page, MEMBER_NAME, "本人");
  await expect(page.getByText("report.png").first()).toBeVisible();
  await expect(page.getByText("1 条")).toBeVisible();
});

test("booking lock, admin approval and visit guide", async ({ page, browser }) => {
  await login(page, PATIENT_USER, PATIENT_PASSWORD);
  await open(page, "/services/");
  await expect(page.getByText("可预约号源")).toBeVisible();

  await page.getByRole("button", { name: /09:00/ }).first().click({ timeout: 30_000 });
  await page.getByRole("button", { name: "锁定号源" }).click();
  await expect(page.getByText(/剩余时间/)).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: "创建订单" }).click();
  await expect(page.getByText(/^order_[0-9a-f]+$/).first()).toBeVisible();
  await page.getByRole("button", { name: "申请支付" }).click();
  await expect(page.getByText(/等待管理员人工确认/)).toBeVisible();

  const adminContext = await browser.newContext();
  const adminPage = await adminContext.newPage();
  await login(adminPage, ADMIN_USER, ADMIN_PASSWORD);
  await open(adminPage, "/admin/");
  await adminPage.getByRole("button", { name: "待审核" }).click();
  await expect(adminPage.getByText("待审核支付申请")).toBeVisible();
  await adminPage.getByRole("button", { name: "批准" }).first().click();
  await expect(adminPage.getByText("暂无待审核申请。")).toBeVisible({ timeout: 30_000 });
  await adminContext.close();

  await expect(page.getByText("已支付", { exact: true })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("就诊指引")).toBeVisible();
  await expect(page.getByText(/示范医院/)).toBeVisible();

  await page.reload();
  await page.waitForLoadState("networkidle");
  await expect(page.getByText("已支付", { exact: true })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("就诊指引")).toBeVisible();
  await expect(page.getByText(/示范医院/)).toBeVisible();
});
