import { expect, test, type APIRequestContext } from "@playwright/test";

import { ADMIN_PASSWORD, ADMIN_USER, API_BASE, PATIENT_PASSWORD, PATIENT_USER } from "./env";

test.describe.configure({ mode: "serial" });

/** 剧本 E2E-02：退号退费闭环
 *  当日挂号 → 收款 → 取消（含退费）→ 双端状态一致（对账无差异）→ 号源可再挂 → 重复取消被拒
 */

async function token(request: APIRequestContext, username: string, password: string): Promise<string> {
  const response = await request.post(API_BASE + "/api/v1/auth/login", {
    data: { username, password },
  });
  expect(response.ok()).toBeTruthy();
  return ((await response.json()) as { access_token: string }).access_token;
}

function auth(t: string): Record<string, string> {
  return { Authorization: "Bearer " + t };
}

function today(): string {
  return new Date().toLocaleDateString("sv-SE");
}

test("E2E-02 退号退费：取消后对账无差异、号源可再挂", async ({ request }) => {
  const patientToken = await token(request, PATIENT_USER, PATIENT_PASSWORD);
  const adminToken = await token(request, ADMIN_USER, ADMIN_PASSWORD);

  const memberResponse = await request.post(API_BASE + "/api/v1/family", {
    headers: auth(patientToken),
    data: {
      name: "退费患者" + Date.now().toString(36),
      relation: "other",
      id_card: "31010119870202" + String(Math.floor(Math.random() * 10000)).padStart(4, "0"),
    },
  });
  expect(memberResponse.ok(), "建档失败").toBeTruthy();
  const memberId = ((await memberResponse.json()) as { member: { id: string } }).member.id;

  const book = async () => {
    const response = await request.post(API_BASE + "/api/v1/booking/appointments", {
      headers: auth(patientToken),
      data: { member_id: memberId, org_id: "101", practitioner_id: "201", service_id: "301", date: today() },
    });
    expect(response.ok(), "挂号失败：" + response.status() + " " + (await response.text())).toBeTruthy();
    return ((await response.json()) as { appointment: { id: string } }).appointment.id;
  };

  // 1) 挂号 → 收款
  const appointmentId = await book();
  const paid = await request.post(API_BASE + "/api/v1/admin/appointments/" + appointmentId + "/simulate-payment", {
    headers: auth(adminToken),
  });
  expect(paid.ok(), "收款失败：" + paid.status()).toBeTruthy();

  const afterPay = await request.get(API_BASE + "/api/v1/booking/appointments/" + appointmentId, {
    headers: auth(patientToken),
  });
  const paidDetail = ((await afterPay.json()) as { appointment: { payment_status: string } }).appointment;
  expect(paidDetail.payment_status).toBe("paid");

  // 2) 取消并退费
  const cancelled = await request.post(API_BASE + "/api/v1/booking/appointments/" + appointmentId + "/cancel", {
    headers: auth(patientToken),
    data: { reason: "患者临时有事", refund: true },
  });
  expect(cancelled.ok(), "取消失败：" + cancelled.status() + " " + (await cancelled.text())).toBeTruthy();

  const afterCancel = await request.get(API_BASE + "/api/v1/booking/appointments/" + appointmentId, {
    headers: auth(patientToken),
  });
  const cancelDetail = ((await afterCancel.json()) as { appointment: { status: string } }).appointment;
  expect(["cancelled", "refunded"]).toContain(cancelDetail.status);

  // 3) 双端一致：主动对账不应再把这笔报成"金额/状态不一致"
  const reconcile = await request.post(API_BASE + "/api/v1/ops/reconciliation/run", {
    headers: auth(adminToken),
    data: { scope: "payment" },
  });
  expect(reconcile.ok(), "对账失败：" + reconcile.status()).toBeTruthy();
  const reconcileBody = (await reconcile.json()) as { findings: Array<{ local_ref?: string; kind?: string }> };
  const mismatched = reconcileBody.findings.filter((item) => item.local_ref === appointmentId);
  expect(mismatched, "取消退费后不应再产生该单的支付差异：" + JSON.stringify(mismatched)).toHaveLength(0);

  // 4) 号源可再挂（退号后同一号源能重新下单）
  const rebooked = await book();
  expect(rebooked).not.toBe(appointmentId);

  // 5) 重复取消走幂等短路：返回现状且不再打 HIS（避免重复退费）
  const again = await request.post(API_BASE + "/api/v1/booking/appointments/" + appointmentId + "/cancel", {
    headers: auth(patientToken),
    data: { reason: "重复取消", refund: true },
  });
  expect(again.status()).toBe(200);
  const againBody = (await again.json()) as { cancel: { already_cancelled?: boolean; status?: string } };
  expect(againBody.cancel.already_cancelled).toBe(true);
  expect(["cancelled", "refunded"]).toContain(againBody.cancel.status);

  // 清理：删掉就诊人，避免污染其它用例（预约记录保留，作为审计对象）
  await request.delete(API_BASE + "/api/v1/family/" + memberId, { headers: auth(patientToken) });
});
