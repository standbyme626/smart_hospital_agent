import path from "node:path";

import { expect, test, type APIRequestContext, type Browser, type Page } from "@playwright/test";

import { REPO_ROOT, ADMIN_PASSWORD, ADMIN_USER, API_BASE, PATIENT_PASSWORD, PATIENT_USER } from "./env";

/** 生成 README 用的界面截图：在隔离环境（独立库 + fake 模型 + mock HIS）里先造一份真实数据，
 *  再按四端逐屏截图。不接触生产库，也不截图浏览器地址栏。 */

const SHOTS = path.join(REPO_ROOT, "docs", "screenshots");

async function token(request: APIRequestContext, username: string, password: string): Promise<string> {
  const response = await request.post(API_BASE + "/api/v1/auth/login", {
    data: { username, password },
  });
  expect(response.ok(), "登录失败：" + response.status()).toBeTruthy();
  return ((await response.json()) as { access_token: string }).access_token;
}

function auth(t: string): Record<string, string> {
  return { Authorization: "Bearer " + t };
}

function today(): string {
  return new Date().toLocaleDateString("sv-SE");
}

async function newPage(browser: Browser, tokenValue: string, width = 1440, height = 900): Promise<Page> {
  const context = await browser.newContext({ viewport: { width, height }, locale: "zh-CN" });
  await context.addInitScript((token: string) => {
    window.localStorage.setItem("sha_token", token);
    window.localStorage.setItem("sha_consent", JSON.stringify({ version: "v", at: new Date().toISOString() }));
  }, tokenValue);
  const page = await context.newPage();
  // 去掉 Next dev 的浮层按钮，截图更干净
  await page.addStyleTag({ content: "nextjs-portal{display:none!important}" }).catch(() => undefined);
  return page;
}

async function shot(page: Page, name: string, url: string, selector?: string): Promise<void> {
  await page.goto(url);
  await page.waitForLoadState("networkidle");
  await page.addStyleTag({ content: "nextjs-portal{display:none!important}" }).catch(() => undefined);
  if (selector) {
    await page.getByRole("button", { name: selector }).first().click();
    await page.waitForTimeout(600);
  }
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(SHOTS, name), fullPage: true });
}

test("生成 README 界面截图", async ({ browser, request }) => {
  test.setTimeout(180_000);
  const patientToken = await token(request, PATIENT_USER, PATIENT_PASSWORD);
  const adminToken = await token(request, ADMIN_USER, ADMIN_PASSWORD);

  // ---------------- 造一份真实数据（全部走真实接口） ----------------
  const memberRes = await request.post(API_BASE + "/api/v1/family", {
    headers: auth(patientToken),
    data: {
      name: "王小明",
      relation: "self",
      is_self: true,
      gender: "男",
      birth_date: "1990-03-12",
      id_card: "31010119900312" + String(Math.floor(Math.random() * 10000)).padStart(4, "0"),
    },
  });
  expect(memberRes.ok(), "建档失败").toBeTruthy();
  const memberId = ((await memberRes.json()) as { member: { id: string } }).member.id;

  await request.post(API_BASE + "/api/v1/previsit", {
    headers: auth(patientToken),
    data: {
      member_id: memberId,
      chief_complaint: "活动后胸闷两天，休息可缓解",
      duration: "两天",
      medications: "无",
      allergies: "青霉素过敏",
      history: "高血压 5 年，规律服药",
      questions: "需要做心电图和心脏彩超吗",
    },
  });

  const apptRes = await request.post(API_BASE + "/api/v1/booking/appointments", {
    headers: auth(patientToken),
    data: { member_id: memberId, org_id: "101", practitioner_id: "201", service_id: "301", date: today() },
  });
  expect(apptRes.ok(), "挂号失败：" + apptRes.status()).toBeTruthy();
  const appointmentId = ((await apptRes.json()) as { appointment: { id: string } }).appointment.id;
  await request.post(API_BASE + "/api/v1/admin/appointments/" + appointmentId + "/simulate-payment", {
    headers: auth(adminToken),
  });

  await request.post(API_BASE + "/api/v1/clinician/receive", {
    headers: auth(adminToken),
    data: { member_id: memberId, note: "门诊接诊" },
  });
  const noteRes = await request.post(API_BASE + "/api/v1/clinician/notes", {
    headers: auth(adminToken),
    data: {
      member_id: memberId,
      chief_complaint: "活动后胸闷两天",
      present_illness: "两天前快步走后出现胸骨后压迫感，休息 3-5 分钟缓解，无放射痛",
      physical_exam: "BP 138/86，HR 78 次/分，律齐，未闻及杂音",
      diagnosis: "稳定型心绞痛（待排）",
    },
  });
  expect(noteRes.ok()).toBeTruthy();
  const noteId = ((await noteRes.json()) as { note: { id: string } }).note.id;

  for (const order of [
    { kind: "medication", name: "阿司匹林肠溶片", dose: "0.1g", usage: "口服 每日一次" },
    { kind: "medication", name: "阿托伐他汀钙片", dose: "20mg", usage: "口服 每晚一次" },
    { kind: "exam", name: "心电图", dose: "静息十二导联", usage: "心功能室" },
  ]) {
    const created = await request.post(API_BASE + "/api/v1/clinician/orders", {
      headers: auth(adminToken),
      data: { member_id: memberId, ...order },
    });
    expect(created.ok()).toBeTruthy();
    const orderId = ((await created.json()) as { order: { id: string } }).order.id;
    await request.post(API_BASE + "/api/v1/clinician/orders/" + orderId + "/sign", {
      headers: auth(adminToken),
    });
  }

  const printRes = await request.post(API_BASE + "/api/v1/clinician/print", {
    headers: auth(adminToken),
    data: { template: "outpatient_note", note_id: noteId },
  });
  expect(printRes.ok()).toBeTruthy();
  const jobId = ((await printRes.json()) as { job: { id: string } }).job.id;
  await request.post(API_BASE + "/api/v1/clinician/print/" + jobId + "/mark-printed", {
    headers: auth(adminToken),
  });

  await request.post(API_BASE + "/api/v1/clinician/notes/" + noteId + "/close-visit", {
    headers: auth(adminToken),
    data: { advice: "两周后复查，胸痛持续不缓解随时急诊", followup_days: 14 },
  });

  await request.post(API_BASE + "/api/v1/feedback", {
    headers: auth(patientToken),
    data: { category: "排队等候", content: "候诊区座位偏少，建议增加座椅与饮水点。" },
  });

  // 通知走真实接口产生：随访安排会写入站内信（短信 / 微信通道未接）
  const dueDays = (days: number) => {
    const date = new Date();
    date.setDate(date.getDate() + days);
    return date.toLocaleDateString("sv-SE");
  };
  for (const seed of [
    { title: "血压复测提醒", days: 1, notes: "早晚各测一次，记录后可上传到健康档案" },
    { title: "心电图复查预约", days: 3, notes: "空腹前往心功能室，携带上次报告" },
    { title: "用药依从性回访", days: 7, notes: "确认阿司匹林与他汀是否规律服用" },
  ]) {
    await request.post(API_BASE + "/api/v1/followups", {
      headers: auth(patientToken),
      data: { title: seed.title, member_id: memberId, due_date: dueDays(seed.days), notes: seed.notes },
    });
  }

  // 读掉最早一条，截图里能同时看到「已读 / 未读」
  const inbox = await request.get(API_BASE + "/api/v1/notifications", { headers: auth(patientToken) });
  if (inbox.ok()) {
    const body = (await inbox.json()) as { notifications?: Array<{ id: number }> };
    const rows = body.notifications || [];
    const oldest = rows[rows.length - 1];
    if (oldest) {
      await request.post(API_BASE + "/api/v1/notifications/" + oldest.id + "/read", {
        headers: auth(patientToken),
      });
    }
  }

  // ---------------- 患者端 ----------------
  const patient = await newPage(browser, patientToken);
  await shot(patient, "01-patient-home.png", "/");
  await shot(patient, "02-patient-services.png", "/services/");
  await shot(patient, "03-patient-reports.png", "/reports/");
  await shot(patient, "04-patient-followups.png", "/followups/");
  await shot(patient, "05-patient-messages.png", "/messages/");

  // ---------------- 医生端 ----------------
  const doctor = await newPage(browser, adminToken);
  await doctor.goto("/clinician/");
  await doctor.waitForLoadState("networkidle");
  await doctor.addStyleTag({ content: "nextjs-portal{display:none!important}" }).catch(() => undefined);
  // 点开今日第一位患者，截图里能看到 360 面板而不是空态
  await doctor.getByRole("button", { name: /王小明/ }).first().click();
  await doctor.waitForTimeout(1200);
  await doctor.screenshot({ path: path.join(SHOTS, "06-clinician-queue.png"), fullPage: true });
  await shot(doctor, "07-clinician-print.png", "/clinician/print/");
  await shot(doctor, "08-clinician-review.png", "/clinician/review/");
  await shot(doctor, "09-clinician-followup.png", "/clinician/followup/");

  // ---------------- 运营端 ----------------
  await shot(doctor, "10-ops-overview.png", "/ops/");
  await shot(doctor, "11-ops-review.png", "/ops/reviews/");

  // ---------------- 管理端 ----------------
  await shot(doctor, "12-platform-overview.png", "/platform/");
  await shot(doctor, "13-platform-knowledge.png", "/platform/knowledge/");
  await shot(doctor, "14-platform-config.png", "/platform/config/");
  await shot(doctor, "15-platform-acceptance.png", "/platform/insight/", "验收自测");
  await shot(doctor, "16-entries.png", "/entries/");

  // ---------------- 移动端 ----------------
  const mobile = await newPage(browser, patientToken, 390, 844);
  await shot(mobile, "17-mobile-home.png", "/");
});
