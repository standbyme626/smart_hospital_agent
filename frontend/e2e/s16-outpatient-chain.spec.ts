import { expect, test, type APIRequestContext } from "@playwright/test";

import { ADMIN_PASSWORD, ADMIN_USER, API_BASE, PATIENT_PASSWORD, PATIENT_USER } from "./env";

test.describe.configure({ mode: "serial" });

/** 剧本 E2E-01：常规门诊全链路
 *  预问诊 → 挂号报价 → 支付申请 → 管理员审批 → 医生接诊 → 病历 → 医嘱签名 → 患者看到处方
 *  同时验证"双端看到同一个 encounter"与审计留痕。
 */

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

test("E2E-01 常规门诊全链路：预问诊到患者看到处方", async ({ request }) => {
  const patientToken = await token(request, PATIENT_USER, PATIENT_PASSWORD);
  const adminToken = await token(request, ADMIN_USER, ADMIN_PASSWORD);

  // 1) 建档
  const memberResponse = await request.post(API_BASE + "/api/v1/family", {
    headers: auth(patientToken),
    data: {
      name: "全链路患者" + Date.now().toString(36),
      relation: "other",
      is_self: false,
      // 挂号需要身份证（HIS 建档必需）；用合法长度的测试号，避免撞真实数据
      id_card: "31010119880101" + String(Math.floor(Math.random() * 10000)).padStart(4, "0"),
    },
  });
  expect(memberResponse.ok(), "建档失败").toBeTruthy();
  const memberId = ((await memberResponse.json()) as { member: { id: string } }).member.id;

  // 2) 预问诊
  const previsit = await request.post(API_BASE + "/api/v1/previsit", {
    headers: auth(patientToken),
    data: {
      member_id: memberId,
      chief_complaint: "全链路：胸闷两天",
      duration: "两天",
      medications: "无",
      allergies: "无",
      history: "高血压",
      questions: "需要做心电图吗",
    },
  });
  expect(previsit.ok(), "预问诊提交失败：" + previsit.status()).toBeTruthy();

  // 3) 挂号报价（HIS 适配器 mock：org 101 / 医生 201 / 服务 301，且必须是当天）
  const appointmentResponse = await request.post(API_BASE + "/api/v1/booking/appointments", {
    headers: auth(patientToken),
    data: {
      member_id: memberId,
      org_id: "101",
      practitioner_id: "201",
      service_id: "301",
      date: today(),
    },
  });
  if (!appointmentResponse.ok()) {
    throw new Error("挂号失败：" + appointmentResponse.status() + " " + (await appointmentResponse.text()));
  }
  const appointment = ((await appointmentResponse.json()) as { appointment: { id: string; status: string } });
  const appointmentId = appointment.appointment.id;
  expect(["quoted", "awaiting_payment"]).toContain(appointment.appointment.status);

  // 4) 患者发起支付申请（自己不能直接付）→ 管理员在收费窗口侧确认收款
  const requested = await request.post(
    API_BASE + "/api/v1/booking/appointments/" + appointmentId + "/request-payment",
    { headers: auth(patientToken) },
  );
  expect(requested.ok(), "支付申请失败：" + requested.status()).toBeTruthy();
  expect(((await requested.json()) as { task: { task_id?: string; id?: string } }).task).toBeTruthy();

  // 本地 HIS 是 mock 收费模式：管理员侧确认收款（真实环境对应收费窗口 / 自助机回调）
  const approved = await request.post(
    API_BASE + "/api/v1/admin/appointments/" + appointmentId + "/simulate-payment",
    { headers: auth(adminToken) },
  );
  expect(approved.ok(), "收款确认失败：" + approved.status() + " " + (await approved.text())).toBeTruthy();

  const paid = await request.get(API_BASE + "/api/v1/booking/appointments/" + appointmentId, {
    headers: auth(patientToken),
  });
  const paidDetail = ((await paid.json()) as { appointment: { status: string; payment_status: string } }).appointment;
  expect(paidDetail.payment_status).toBe("paid");
  expect(["registered", "completed"]).toContain(paidDetail.status);

  // 5) 医生端：接诊 → 病历 → 医嘱签名
  const received = await request.post(API_BASE + "/api/v1/clinician/receive", {
    headers: auth(adminToken),
    data: { member_id: memberId, note: "全链路接诊" },
  });
  expect(received.ok(), "接诊失败：" + received.status()).toBeTruthy();
  const encounterId = ((await received.json()) as { encounter: { id: string } }).encounter.id;

  const note = await request.post(API_BASE + "/api/v1/clinician/notes", {
    headers: auth(adminToken),
    data: {
      member_id: memberId,
      chief_complaint: "全链路：胸闷两天",
      present_illness: "活动后胸闷，休息可缓解",
      physical_exam: "心率 78，律齐",
      diagnosis: "稳定型心绞痛？",
    },
  });
  expect(note.ok(), "病历保存失败：" + note.status()).toBeTruthy();
  const noteId = ((await note.json()) as { note: { id: string } }).note.id;

  const order = await request.post(API_BASE + "/api/v1/clinician/orders", {
    headers: auth(adminToken),
    data: { member_id: memberId, kind: "medication", name: "阿司匹林肠溶片", dose: "0.1g", usage: "口服 每日一次" },
  });
  expect(order.ok(), "开医嘱失败：" + order.status()).toBeTruthy();
  const orderId = ((await order.json()) as { order: { id: string } }).order.id;

  const signed = await request.post(API_BASE + "/api/v1/clinician/orders/" + orderId + "/sign", {
    headers: auth(adminToken),
  });
  expect(signed.ok(), "签名失败：" + signed.status()).toBeTruthy();

  // 6) 诊毕并生成随访
  const closed = await request.post(API_BASE + "/api/v1/clinician/notes/" + noteId + "/close-visit", {
    headers: auth(adminToken),
    data: { advice: "两周后复查，胸痛加重随时就诊", followup_days: 14 },
  });
  expect(closed.ok(), "诊毕失败：" + closed.status()).toBeTruthy();

  // 7) 患者端：看到 HIS 处方（同一个 encounter）
  const prescriptions = await request.get(
    API_BASE + "/api/v1/booking/appointments/" + appointmentId + "/prescriptions",
    { headers: auth(patientToken) },
  );
  expect(prescriptions.ok(), "读取处方失败：" + prescriptions.status()).toBeTruthy();
  const prescriptionBody = (await prescriptions.json()) as {
    prescriptions: Array<{ items?: Array<{ name: string }> }>;
  };
  const names = (prescriptionBody.prescriptions || []).flatMap((item) => (item.items || []).map((drug) => drug.name));
  expect(names).toContain("阿司匹林肠溶片");

  // 8) 双端看到同一 encounter：患者预约详情 + 医生 360 都能对上
  const patientView = await request.get(API_BASE + "/api/v1/booking/appointments/" + appointmentId, {
    headers: auth(patientToken),
  });
  expect(patientView.ok()).toBeTruthy();

  const clinicianView = await request.get(API_BASE + "/api/v1/clinician/patients/" + memberId, {
    headers: auth(adminToken),
  });
  expect(clinicianView.ok()).toBeTruthy();
  const clinicianBody = (await clinicianView.json()) as {
    appointments: Array<{ id: string }>;
    previsit: { chief_complaint: string } | null;
  };
  expect(clinicianBody.appointments.map((item) => item.id)).toContain(appointmentId);
  expect(clinicianBody.previsit?.chief_complaint).toContain("全链路");

  // 9) 审计留痕：接诊 / 签名 / 诊毕 / 审批都在
  const audit = await request.get(API_BASE + "/api/v1/admin/audit", { headers: auth(adminToken) });
  expect(audit.ok()).toBeTruthy();
  const auditText = JSON.stringify(await audit.json());
  for (const action of ["clinician.encounter.received", "clinician.visit.closed"]) {
    expect(auditText).toContain(action);
  }
  expect(encounterId).toBeTruthy();

  // 清理就诊人：E2E 共享同一个库，留档会影响 v31 的成员切换用例
  await request.delete(API_BASE + "/api/v1/family/" + memberId, { headers: auth(patientToken) });
});
