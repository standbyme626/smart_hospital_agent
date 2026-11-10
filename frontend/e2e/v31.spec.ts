import path from "node:path";

import { expect, test, type Page } from "@playwright/test";

import { ADMIN_PASSWORD, ADMIN_USER, API_BASE, PATIENT_PASSWORD, PATIENT_USER } from "./env";

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
  await page.getByRole("checkbox", { name: /我已知悉并同意/ }).check();
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
  await expect(page.getByText(/\d+ 条/).first()).toBeVisible();
});

test("HIS booking quote, cashier payment and visit guide", async ({ page, browser, request }) => {
  const auth = await request.post(`${API_BASE}/api/v1/auth/login`, {
    data: { username: PATIENT_USER, password: PATIENT_PASSWORD },
  });
  expect(auth.ok()).toBeTruthy();
  const token = ((await auth.json()) as { access_token: string }).access_token;
  const family = await request.get(`${API_BASE}/api/v1/family`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  expect(family.ok()).toBeTruthy();
  const members = ((await family.json()) as { members: Array<Record<string, unknown>> }).members;
  const selfMember = members.find((item) => item.is_self) as Record<string, unknown> | undefined;
  expect(selfMember).toBeTruthy();
  const patched = await request.patch(`${API_BASE}/api/v1/family/${selfMember!.id}`, {
    headers: { Authorization: `Bearer ${token}` },
    data: {
      name: selfMember!.name,
      relation: "self",
      is_self: true,
      id_card: `31010119880101${String(Math.floor(Math.random() * 10000)).padStart(4, "0")}`,
      birth_date: "1988-01-01",
      gender: "male",
    },
  });
  expect(patched.ok()).toBeTruthy();

  await login(page, PATIENT_USER, PATIENT_PASSWORD);
  await open(page, "/services/");
  await expect(page.getByText("HIS 报价 + 医院收费")).toBeVisible();
  const bookingDepartment = process.env.E2E_BOOKING_DEPARTMENT || "心血管内科";
  const bookingDoctor = process.env.E2E_BOOKING_DOCTOR || "张医生";
  const bookingService = process.env.E2E_BOOKING_SERVICE || "心血管内科普通门诊";
  const departmentSelect = page.locator("select").first();
  if (await departmentSelect.count()) {
    await departmentSelect.selectOption({ label: bookingDepartment }).catch(() => {});
  }
  const doctorButton = page.getByRole("button", { name: `${bookingDoctor} 预约挂号` });
  await expect(doctorButton).toBeVisible({ timeout: 30_000 });
  await doctorButton.click();
  await page.getByRole("button", { name: new RegExp(bookingService) }).first().click();
  await page.getByRole("button", { name: /提交挂号（获取报价）/ }).click();
  await expect(page.getByText("报价单号：")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("去缴费 / 申请支付")).toBeVisible();

  await page.getByRole("button", { name: /去缴费 \/ 申请支付/ }).click();
  await expect(page.getByText(/收费任务已创建/)).toBeVisible({ timeout: 30_000 });

  const adminContext = await browser.newContext();
  const adminPage = await adminContext.newPage();
  await login(adminPage, ADMIN_USER, ADMIN_PASSWORD);
  await open(adminPage, "/admin/");
  await adminPage.getByRole("button", { name: /收费任务/ }).click();
  await expect(adminPage.getByText("收费任务（待处理）")).toBeVisible({ timeout: 30_000 });
  adminPage.once("dialog", (dialog) => dialog.accept());
  await adminPage.getByRole("button", { name: /模拟支付（演示）/ }).first().click();
  await expect(adminPage.getByText(/状态将随 HIS 同步/)).toBeVisible({ timeout: 30_000 });
  await adminContext.close();

  await expect(page.getByText("已挂号", { exact: true }).first()).toBeVisible({ timeout: 45_000 });
  await expect(page.getByText("就诊指引")).toBeVisible();
  await expect(page.getByText(/请提前 20 分钟到院/)).toBeVisible();

  await page.reload();
  await page.waitForLoadState("networkidle");
  await expect(page.getByText("已挂号", { exact: true }).first()).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("就诊指引")).toBeVisible();
});

function ageFor(year: number, month: number, day: number): number {
  const now = new Date();
  let age = now.getFullYear() - year;
  if (now.getMonth() + 1 < month || (now.getMonth() + 1 === month && now.getDate() < day)) age -= 1;
  return age;
}

test("consult age follows the selected member's birth date", async ({ page, request }) => {
  const auth = await request.post(`${API_BASE}/api/v1/auth/login`, {
    data: { username: PATIENT_USER, password: PATIENT_PASSWORD },
  });
  expect(auth.ok()).toBeTruthy();
  const token = ((await auth.json()) as { access_token: string }).access_token;
  const family = await request.get(`${API_BASE}/api/v1/family`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  expect(family.ok()).toBeTruthy();
  const selfMember = ((await family.json()) as { members: Array<Record<string, unknown>> }).members.find(
    (item) => item.is_self,
  ) as Record<string, unknown> | undefined;
  expect(selfMember).toBeTruthy();
  const reset = await request.patch(`${API_BASE}/api/v1/family/${selfMember!.id}`, {
    headers: { Authorization: `Bearer ${token}` },
    data: {
      name: selfMember!.name,
      relation: "self",
      is_self: true,
      birth_date: null,
      gender: null,
    },
  });
  expect(reset.ok()).toBeTruthy();
  const created = await request.post(`${API_BASE}/api/v1/family`, {
    headers: { Authorization: `Bearer ${token}` },
    data: { name: "年龄测试家人", relation: "子女", birth_date: "2018-06-01" },
  });
  expect(created.ok()).toBeTruthy();
  const invalid = await request.post(`${API_BASE}/api/v1/family`, {
    headers: { Authorization: `Bearer ${token}` },
    data: { name: "无效生日家人", relation: "父母", birth_date: "1994-13-40" },
  });
  expect(invalid.ok()).toBeTruthy();

  await login(page, PATIENT_USER, PATIENT_PASSWORD);
  await open(page, "/consult/");
  await expect(page.getByText("这次为谁咨询？")).toBeVisible();
  const memberCard = page.locator(".card").first();
  const ageInput = memberCard.locator('label:has-text("年龄") input');

  await memberCard.getByRole("button", { name: /^年龄测试家人/ }).click();
  await expect(ageInput).toHaveValue(String(ageFor(2018, 6, 1)));
  await expect(memberCard.getByText("来自家庭档案出生日期")).toBeVisible();

  await memberCard.getByRole("button", { name: /^无效生日家人/ }).click();
  await expect(memberCard.getByText("档案出生日期无效，请修改；本次可手动填写年龄")).toBeVisible();
  await ageInput.fill("35");
  await expect(ageInput).toHaveValue("35");

  await memberCard.getByRole("button", { name: /^本人$/ }).click();
  await expect(ageInput).toHaveValue("");
});
