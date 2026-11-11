/** 医生端收尾客户端：文书打印 / 诊毕随访 / 审方聚合。 */
import { API_BASE, ApiError, getToken } from "./api";

export type PrintTemplate = { key: string; name: string; description: string };

export type PrintJobSummary = {
  id: string;
  template: string;
  template_name: string;
  note_id: string | null;
  member_id: string;
  doctor: string | null;
  status: "queued" | "printed" | string;
  created_at: string | null;
  printed_at: string | null;
};

export type PrintJobDetail = {
  job: {
    id: string;
    template: string;
    member_id: string;
    doctor: string | null;
    status: string;
    created_at: string | null;
    printed_at: string | null;
  };
  content: string;
};

export type FollowupPlan = {
  id: string;
  note_id: string;
  member_id: string;
  member_name: string | null;
  doctor: string | null;
  advice: string | null;
  followup_days: number;
  followup_id: string | null;
  title: string | null;
  due_date: string | null;
  status: string | null;
  created_at: string | null;
};

export type CloseVisitResult = {
  note: { id: string; status: string };
  followup: {
    id: string;
    member_id: string;
    title: string;
    due_date: string;
    status: string;
    notes: string | null;
  };
  closure: { id: string; followup_days: number; created_at: string | null };
};

export type PrescriptionReview = {
  id: string;
  member_id: string;
  member_name: string | null;
  kind: string;
  name: string;
  dose: string | null;
  usage: string | null;
  doctor: string | null;
  review_status: "pending" | "approved" | "rejected" | string;
  review_note: string | null;
  signed_at: string | null;
};

const ROOT = "/api/v1/clinician";

async function clinicianFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", "Bearer " + token);
  if (init.body) headers.set("Content-Type", "application/json");

  let response: Response;
  try {
    response = await fetch(API_BASE + path, { ...init, headers });
  } catch {
    throw new ApiError(0, "网络连接失败，请检查网络后重试");
  }
  if (!response.ok) {
    let message = "";
    try {
      const data = (await response.json()) as { detail?: unknown };
      const detail = data?.detail;
      message = typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : "";
    } catch {
      message = "";
    }
    throw new ApiError(response.status, message || "请求失败（" + response.status + "）");
  }
  return (await response.json()) as T;
}

export const getPrintTemplates = () =>
  clinicianFetch<{ items: PrintTemplate[] }>(ROOT + "/print/templates");

export const createPrintJob = (payload: { template: string; note_id?: string; member_id?: string }) =>
  clinicianFetch<{ job: PrintJobSummary; content: string }>(ROOT + "/print", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const listPrintJobs = (memberId?: string) =>
  clinicianFetch<{ items: PrintJobSummary[] }>(
    ROOT + "/print?limit=20" + (memberId ? "&member_id=" + encodeURIComponent(memberId) : ""),
  );

export const getPrintJob = (jobId: string) =>
  clinicianFetch<PrintJobDetail>(ROOT + "/print/" + encodeURIComponent(jobId));

export const markPrinted = (jobId: string) =>
  clinicianFetch<{ job: { id: string; status: string; printed_at: string | null } }>(
    ROOT + "/print/" + encodeURIComponent(jobId) + "/mark-printed",
    { method: "POST" },
  );

export const closeVisit = (noteId: string, payload: { advice?: string; followup_days: number }) =>
  clinicianFetch<CloseVisitResult>(ROOT + "/notes/" + encodeURIComponent(noteId) + "/close-visit", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const getFollowupPlans = () =>
  clinicianFetch<{ items: FollowupPlan[] }>(ROOT + "/followup-plans?limit=20");

export const getPrescriptionReviews = (status?: string) =>
  clinicianFetch<{ items: PrescriptionReview[]; summary: Record<string, number>; his_note: string }>(
    ROOT + "/prescription-reviews" + (status ? "?review_status=" + encodeURIComponent(status) : ""),
  );

export const reviewOrder = (orderId: string, status: "approved" | "rejected", note?: string) =>
  clinicianFetch<{ order: Record<string, unknown> }>(
    ROOT + "/orders/" + encodeURIComponent(orderId) + "/review",
    { method: "POST", body: JSON.stringify({ status, note }) },
  );
