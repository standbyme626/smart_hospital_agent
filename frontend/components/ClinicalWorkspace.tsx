"use client";

import { useCallback, useEffect, useState } from "react";
import { CheckCircle2, ClipboardList, FlaskConical, Pill, Printer, Stethoscope } from "lucide-react";
import {
  completeClinicalNote,
  createClinicalNote,
  createClinicalOrder,
  createConsultation,
  getClinicalNotes,
  getClinicalOrders,
  getConsultations,
  getLabResults,
  reviewClinicalOrder,
  signClinicalOrder,
  type ClinicalNote,
  type ClinicalOrder,
  type Consultation,
  type LabResult,
} from "@/lib/api";

type Tab = "note" | "order" | "lab" | "consult";

const TABS: Array<{ key: Tab; label: string; icon: typeof ClipboardList }> = [
  { key: "note", label: "病历", icon: ClipboardList },
  { key: "order", label: "医嘱/处方", icon: Pill },
  { key: "lab", label: "检验", icon: FlaskConical },
  { key: "consult", label: "会诊", icon: Stethoscope },
];

export function ClinicalWorkspace({ memberId, writeEnabled }: { memberId: string; writeEnabled: boolean }) {
  const [tab, setTab] = useState<Tab>("note");
  const [notes, setNotes] = useState<ClinicalNote[]>([]);
  const [orders, setOrders] = useState<ClinicalOrder[]>([]);
  const [labs, setLabs] = useState<LabResult[]>([]);
  const [consults, setConsults] = useState<Consultation[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [chief, setChief] = useState("");
  const [present, setPresent] = useState("");
  const [exam, setExam] = useState("");
  const [diagnosis, setDiagnosis] = useState("");

  const [orderKind, setOrderKind] = useState("medication");
  const [orderName, setOrderName] = useState("");
  const [orderDose, setOrderDose] = useState("");
  const [orderUsage, setOrderUsage] = useState("");

  const [consultDept, setConsultDept] = useState("神经内科");
  const [consultReason, setConsultReason] = useState("");

  const loadAll = useCallback(async () => {
    setError(null);
    try {
      const [n, o, l, c] = await Promise.all([
        getClinicalNotes(memberId).catch(() => []),
        getClinicalOrders(memberId).catch(() => []),
        getLabResults(memberId).catch(() => []),
        getConsultations(memberId).catch(() => []),
      ]);
      setNotes(n);
      setOrders(o);
      setLabs(l);
      setConsults(c);
    } catch {
      setError("加载失败，请稍后重试");
    }
  }, [memberId]);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  async function saveNote() {
    if (!chief.trim()) return;
    setBusy("note");
    try {
      await createClinicalNote({
        member_id: memberId,
        chief_complaint: chief.trim(),
        present_illness: present.trim() || undefined,
        physical_exam: exam.trim() || undefined,
        diagnosis: diagnosis.trim() || undefined,
      });
      setChief("");
      setPresent("");
      setExam("");
      setDiagnosis("");
      await loadAll();
    } catch {
      setError("保存病历失败");
    } finally {
      setBusy(null);
    }
  }

  async function completeNote(id: string) {
    setBusy(id);
    try {
      await completeClinicalNote(id);
      await loadAll();
    } finally {
      setBusy(null);
    }
  }

  async function addOrder() {
    if (!orderName.trim()) return;
    setBusy("order");
    try {
      await createClinicalOrder({
        member_id: memberId,
        kind: orderKind,
        name: orderName.trim(),
        dose: orderDose.trim() || undefined,
        usage: orderUsage.trim() || undefined,
      });
      setOrderName("");
      setOrderDose("");
      setOrderUsage("");
      await loadAll();
    } catch {
      setError("添加医嘱失败");
    } finally {
      setBusy(null);
    }
  }

  async function signOrder(id: string) {
    setBusy(id);
    setError(null);
    try {
      await signClinicalOrder(id);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "签名失败");
    } finally {
      setBusy(null);
    }
  }

  async function reviewOrder(id: string, status: string) {
    setBusy(id);
    try {
      await reviewClinicalOrder(id, status, status === "approved" ? "审方通过" : "需调整");
      await loadAll();
    } finally {
      setBusy(null);
    }
  }

  async function addConsult() {
    if (!consultDept.trim()) return;
    setBusy("consult");
    try {
      await createConsultation({
        member_id: memberId,
        target_department: consultDept.trim(),
        reason: consultReason.trim() || undefined,
      });
      setConsultReason("");
      await loadAll();
    } finally {
      setBusy(null);
    }
  }

  return (
    <section className="card p-5">
      <div className="flex flex-wrap gap-2">
        {TABS.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.key}
              onClick={() => setTab(item.key)}
              className={tab === item.key ? "chip flex items-center gap-1 border-primary bg-primary text-white" : "chip flex items-center gap-1 text-muted"}
            >
              <Icon size={12} /> {item.label}
            </button>
          );
        })}
      </div>
      {error ? <p className="mt-2 text-[12px] text-rose-600">{error}</p> : null}

      {tab === "note" ? (
        <div className="mt-3 space-y-3">
          <div className="grid gap-2 sm:grid-cols-2">
            <input className="input" value={chief} onChange={(e) => setChief(e.target.value)} placeholder="主诉（必填）" />
            <input className="input" value={diagnosis} onChange={(e) => setDiagnosis(e.target.value)} placeholder="初步诊断" />
            <input className="input sm:col-span-2" value={present} onChange={(e) => setPresent(e.target.value)} placeholder="现病史" />
            <input className="input sm:col-span-2" value={exam} onChange={(e) => setExam(e.target.value)} placeholder="查体" />
          </div>
          <button onClick={saveNote} disabled={busy === "note" || !chief.trim()} className="btn-primary min-h-9 px-4 text-[12px] disabled:opacity-50">
            保存病历
          </button>
          <div className="space-y-2">
            {notes.length === 0 ? <p className="rounded-2xl bg-canvas p-3 text-[12px] text-muted">还没有病历</p> : notes.map((note) => (
              <div key={note.id} className="rounded-2xl border border-line p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="badge">{note.status === "completed" ? "已诊毕" : "草稿"}</span>
                  <span className="text-[13px] font-medium text-ink">{note.chief_complaint}</span>
                  <span className="ml-auto text-[11px] text-muted">{note.created_at?.slice(0, 10)}</span>
                </div>
                {note.diagnosis ? <p className="mt-1 text-[12px] text-muted">诊断：{note.diagnosis}</p> : null}
                <div className="mt-2 flex items-center gap-2">
                  {note.status === "draft" ? (
                    <button onClick={() => completeNote(note.id)} disabled={busy === note.id} className="text-[12px] font-medium text-primary disabled:text-muted">
                      诊毕
                    </button>
                  ) : null}
                  <button onClick={() => window.print()} className="ml-auto flex items-center gap-1 text-[12px] text-muted hover:text-primary">
                    <Printer size={12} /> 打印
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {tab === "order" ? (
        <div className="mt-3 space-y-3">
          <div className="grid gap-2 sm:grid-cols-4">
            <select className="input" value={orderKind} onChange={(e) => setOrderKind(e.target.value)}>
              <option value="medication">药品</option>
              <option value="exam">检查</option>
              <option value="lab">检验</option>
            </select>
            <input className="input" value={orderName} onChange={(e) => setOrderName(e.target.value)} placeholder="名称" />
            <input className="input" value={orderDose} onChange={(e) => setOrderDose(e.target.value)} placeholder="剂量/频次" />
            <input className="input" value={orderUsage} onChange={(e) => setOrderUsage(e.target.value)} placeholder="用法" />
          </div>
          <div className="flex items-center gap-2">
            <button onClick={addOrder} disabled={busy === "order" || !orderName.trim()} className="btn-primary min-h-9 px-4 text-[12px] disabled:opacity-50">添加草稿</button>
            <span className="text-[11px] text-muted">{writeEnabled ? "签名将写入 HIS" : "签名未启用：需要医生 HIS 账号透传"}</span>
          </div>
          <div className="space-y-2">
            {orders.length === 0 ? <p className="rounded-2xl bg-canvas p-3 text-[12px] text-muted">还没有医嘱</p> : orders.map((order) => (
              <div key={order.id} className="flex flex-wrap items-center gap-2 rounded-2xl border border-line p-3">
                <span className="badge">{order.kind === "medication" ? "药品" : order.kind === "exam" ? "检查" : "检验"}</span>
                <span className="text-[13px] font-medium text-ink">{order.name}{order.dose ? " · " + order.dose : ""}</span>
                <span className="text-[11px] text-muted">{order.usage || ""}</span>
                <span className={order.status === "signed" ? "badge bg-emerald-50 text-emerald-700" : "badge bg-amber-50 text-amber-700"}>
                  {order.status === "signed" ? "已签名" : "草稿"}
                </span>
                {order.status === "signed" ? <span className="badge">审方：{order.review_status}</span> : null}
                <div className="ml-auto flex items-center gap-2">
                  {order.status === "draft" ? (
                    <button onClick={() => signOrder(order.id)} disabled={!writeEnabled || busy === order.id} title={writeEnabled ? "" : "HIS 写入未启用"} className="text-[12px] font-medium text-primary disabled:text-muted">
                      签名
                    </button>
                  ) : order.review_status === "pending" ? (
                    <>
                      <button onClick={() => reviewOrder(order.id, "approved")} className="text-[12px] text-emerald-600">审方通过</button>
                      <button onClick={() => reviewOrder(order.id, "rejected")} className="text-[12px] text-rose-600">驳回</button>
                    </>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {tab === "lab" ? (
        <div className="mt-3 overflow-hidden rounded-2xl border border-line">
          <table className="w-full text-left text-[12px]">
            <thead className="bg-canvas text-muted">
              <tr><th className="px-3 py-2">项目</th><th className="px-3 py-2">结果</th><th className="px-3 py-2">参考</th><th className="px-3 py-2">时间</th></tr>
            </thead>
            <tbody>
              {labs.length === 0 ? (
                <tr><td className="px-3 py-3 text-muted" colSpan={4}>未记录</td></tr>
              ) : labs.map((lab) => (
                <tr key={lab.id} className="border-t border-line">
                  <td className="px-3 py-2 text-ink">{lab.name}</td>
                  <td className={lab.abnormal ? "px-3 py-2 text-rose-600" : "px-3 py-2 text-ink"}>{lab.result_value}</td>
                  <td className="px-3 py-2 text-muted">{lab.reference_range}</td>
                  <td className="px-3 py-2 text-muted">{lab.measured_at?.slice(0, 10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      {tab === "consult" ? (
        <div className="mt-3 space-y-3">
          <div className="grid gap-2 sm:grid-cols-[180px_1fr]">
            <input className="input" value={consultDept} onChange={(e) => setConsultDept(e.target.value)} placeholder="目标科室" />
            <input className="input" value={consultReason} onChange={(e) => setConsultReason(e.target.value)} placeholder="会诊理由" />
          </div>
          <button onClick={addConsult} disabled={busy === "consult"} className="btn-primary min-h-9 px-4 text-[12px] disabled:opacity-50">提交会诊</button>
          <div className="space-y-2">
            {consults.length === 0 ? <p className="rounded-2xl bg-canvas p-3 text-[12px] text-muted">还没有会诊记录</p> : consults.map((item) => (
              <div key={item.id} className="rounded-2xl bg-canvas p-3">
                <div className="flex items-center gap-2">
                  <CheckCircle2 size={13} className="text-primary" />
                  <span className="text-[13px] font-medium text-ink">{item.target_department}</span>
                  <span className="badge">{item.status === "requested" ? "待回复" : item.status}</span>
                  <span className="ml-auto text-[11px] text-muted">{item.created_at?.slice(0, 10)}</span>
                </div>
                {item.reason ? <p className="mt-1 text-[12px] text-muted">{item.reason}</p> : null}
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}
