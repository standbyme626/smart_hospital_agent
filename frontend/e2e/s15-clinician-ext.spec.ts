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

async function apiToken(
  request: APIRequestContext,
  username: string,
  password: string,
): Promise<string> {
  const response = await request.post(API_BASE + "/api/v1/auth/login", {
    data: { username, password },
  });
  expect(response.ok(), "登录接口失败：" + response.status()).toBeTruthy();
  return ((await response.json()) as { access_token: string }).access_token;
}

test("医生端权限：患者访问文书打印被拒绝", async ({ page }) => {
  await signIn(page, page.request, PATIENT_USER, PATIENT_PASSWORD);
  await open(page, "/clinician/print/");
  await expect(page.getByText("需要医生权限")).toBeVisible();
});

test("文书打印：生成门诊病历打印任务并留痕", async ({ page, request }) => {
  const patientToken = await apiToken(request, PATIENT_USER, PATIENT_PASSWORD);
  const doctorToken = await apiToken(request, ADMIN_USER, ADMIN_PASSWORD);

  const memberResponse = await request.post(API_BASE + "/api/v1/family", {
    headers: { Authorization: "Bearer " + patientToken },
    data: { name: "E2E文书患者", relation: "self", is_self: true },
  });
  expect(memberResponse.ok(), "创建就诊人失败：" + memberResponse.status()).toBeTruthy();
  const memberId = ((await memberResponse.json()) as { member: { id: string } }).member.id;

  const noteResponse = await request.post(API_BASE + "/api/v1/clinician/notes", {
    headers: { Authorization: "Bearer " + doctorToken },
    data: {
      member_id: memberId,
      chief_complaint: "E2E 主诉：咳嗽一周",
      diagnosis: "急性支气管炎",
      present_illness: "干咳无痰",
      physical_exam: "咽部充血",
    },
  });
  expect(noteResponse.ok(), "创建病历失败：" + noteResponse.status()).toBeTruthy();

  await signIn(page, page.request, ADMIN_USER, ADMIN_PASSWORD);
  await open(page, "/clinician/print/");
  await expect(page.getByRole("heading", { name: "文书打印" })).toBeVisible();

  await page.getByLabel("手动输入就诊人 ID").fill(memberId);
  const noteSelect = page.getByLabel("病历（门诊病历必选）");
  await expect(noteSelect.locator("option", { hasText: "E2E 主诉" })).toHaveCount(1, { timeout: 30_000 });
  await noteSelect.selectOption({ index: 1 });
  await page.getByRole("button", { name: "生成打印任务" }).click();

  await expect(page.getByText("打印预览")).toBeVisible({ timeout: 30_000 });
  const preview = page.locator("pre").first();
  await expect(preview).toContainText("E2E 主诉：咳嗽一周");
  await expect(preview).toContainText("急性支气管炎");
  await expect(preview).toContainText("草稿（未诊毕）");

  await page.getByRole("button", { name: "标记已打印" }).click();
  await expect(page.getByText(/已标记打印完成/)).toBeVisible();
});

test("诊毕与随访：页面可进入且给出随访规则", async ({ page }) => {
  await signIn(page, page.request, ADMIN_USER, ADMIN_PASSWORD);
  await open(page, "/clinician/followup/");
  await expect(page.getByRole("heading", { name: "诊毕与随访" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "随访计划" })).toBeVisible();
  await expect(page.getByText(/重复诊毕会被拒绝/)).toBeVisible();
});

test("药师审方：标注 HIS 口径并可筛选", async ({ page }) => {
  await signIn(page, page.request, ADMIN_USER, ADMIN_PASSWORD);
  await open(page, "/clinician/review/");
  await expect(page.getByRole("heading", { name: "药师审方" })).toBeVisible();
  await expect(page.getByText(/HIS 药师审方记录待院方开通/)).toBeVisible();
  for (const name of [/待审/, /通过/, /退回/, /全部/]) {
    await expect(page.getByRole("button", { name })).toBeVisible();
  }
});
