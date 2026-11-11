"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { Activity, ArrowLeft, CalendarCheck2, ClipboardList, FileText, PenLine, Pill, ShieldAlert, Stethoscope } from "lucide-react";
import { ClinicalWorkspace } from "@/components/ClinicalWorkspace";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { ClinicianNav } from "@/components/ClinicianNav";
import {
  createClinicianDraft,
  getClinicianConfig,
  getClinicianDrafts,
  getClinicianPatient,
  getClinicianQueue,
  signClinicianDraft,
  type ClinicianDraft,
  type ClinicianPatient,
  type ClinicianQueueItem,
} from "@/lib/api";
import { formatDate, relativeTime } from "@/lib/format";

function ageOf(birth?: string | null): string {
  if (!birth) return "";
  const year = Number(birth.slice(0, 4));
  if (!year) return "";
  return String(new Date().getFullYear() - year) + " 岁";
}

export default function ClinicianPage() {
  const [queue, setQueue] = useState<ClinicianQueueItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [denied, setDenied] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<ClinicianPatient | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState(false);
  const [writeEnabled, setWriteEnabled] = useState(false);
  const [drafts, setDrafts] = useState<ClinicianDraft[]>([]);
  const [draftText, setDraftText] = useState("");
  const [draftSaving, setDraftSaving] = useState(false);
  const [signingId, setSigningId] = useState<string | null>(null);
  const [draftError, setDraftError] = useState<string | null>(null);

  const loadQueue = useCallback(async () => {
    setLoading(true);
    setFailed(false);
    setDenied(false);
    try {
      setQueue(await getClinicianQueue());
    } catch (error) {
      const status = (error as { status?: number } | null)?.status;
      if (status === 401 || status === 403) setDenied(true);
      else setFailed(true);
      setQueue([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadQueue();
  }, [loadQueue]);

  useEffect(() => {
    getClinicianConfig()
      .then((config) => setWriteEnabled(config.write_enabled))
      .catch(() => setWriteEnabled(false));
  }, []);

  async function openPatient(memberId: string) {
    setSelected(memberId);
    setDetail(null);
    setDetailError(false);
    setDetailLoading(true);
    setDrafts([]);
    setDraftText("");
    setDraftError(null);
    try {
      const [patient, draftList] = await Promise.all([
        getClinicianPatient(memberId),
        getClinicianDrafts(memberId).catch(() => [] as ClinicianDraft[]),
      ]);
      setDetail(patient);
      setDrafts(draftList);
    } catch {
      setDetailError(true);
    } finally {
      setDetailLoading(false);
    }
  }

  async function saveDraft() {
    if (!selected || !draftText.trim()) return;
    setDraftSaving(true);
    setDraftError(null);
    try {
      const draft = await createClinicianDraft({ member_id: selected, content: draftText.trim() });
      setDrafts((prev) => [draft, ...prev]);
      setDraftText("");
    } catch {
      setDraftError("保存草稿失败，请稍后再试");
    } finally {
      setDraftSaving(false);
    }
  }

  async function signDraft(draft: ClinicianDraft) {
    setSigningId(draft.id);
    setDraftError(null);
    try {
      const signed = await signClinicianDraft(draft.id);
      setDrafts((prev) => prev.map((item) => (item.id === signed.id ? signed : item)));
    } catch (error) {
      setDraftError(error instanceof Error ? error.message : "签名失败，请稍后再试");
    } finally {
      setSigningId(null);
    }
  }

  return (
    <div className="min-h-dvh bg-canvas">
      <header className="sticky top-0 z-20 border-b border-line bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center gap-3 px-4 py-3">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary-soft text-primary">
            <Stethoscope size={17} />
          </span>
          <div className="min-w-0">
            <p className="text-[15px] font-semibold text-ink">医生工作台</p>
            <p className="text-[11px] text-muted">今日队列 / 患者 360 / 病历 / 医嘱 / 检验 / 会诊；文书打印、诊毕随访、审方在顶部导航</p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <SourceBadge live />
            <Link href="/" className="inline-flex min-h-9 items-center gap-1 rounded-full bg-white px-3 text-[12px] text-muted ring-1 ring-line hover:text-primary">
              <ArrowLeft size={13} /> 患者端
            </Link>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-4 px-4 py-5">
        <ClinicianNav />
        <OfflineBanner />

        {denied ? (
          <div className="card p-3">
            <StateBlock kind="denied" title="当前账号无医生权限" description="医生端只对 doctor / admin 角色开放；请用医生账号登录" />
            <div className="mt-2 text-center">
              <Link href="/login/" className="text-[12px] font-medium text-primary">去登录</Link>
            </div>
          </div>
        ) : (
          <div className="grid gap-4 lg:grid-cols-[360px_1fr]">
            <section className="card overflow-hidden">
              <div className="flex items-center justify-between px-4 py-3">
                <h2 className="section-title">今日患者</h2>
                <span className="text-[11px] text-muted">{loading || failed ? "—" : queue.length + " 人"}</span>
              </div>
              <div className="border-t border-line">
                {loading ? (
                  <div className="space-y-2 p-3">
                    <div className="h-14 animate-pulse rounded-2xl bg-canvas" />
                    <div className="h-14 animate-pulse rounded-2xl bg-canvas" />
                  </div>
                ) : failed ? (
                  <div className="p-3"><StateBlock kind="error" onRetry={loadQueue} /></div>
                ) : queue.length === 0 ? (
                  <div className="p-3"><StateBlock kind="empty" title="今天还没有挂号" description="有患者挂号后会自动出现在这里" /></div>
                ) : (
                  <ul className="divide-y divide-line">
                    {queue.map((item) => (
                      <li key={item.id}>
                        <button
                          onClick={() => openPatient(item.member_id)}
                          className={
                            selected === item.member_id
                              ? "flex w-full items-center gap-3 bg-primary-soft/50 px-4 py-3 text-left"
                              : "flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-primary-soft/30"
                          }
                        >
                          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-canvas text-[12px] font-medium text-muted">
                            {(item.member_name || "患").slice(0, 1)}
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className="flex items-center gap-2">
                              <span className="truncate text-[13px] font-medium text-ink">{item.member_name || item.member_id}</span>
                              {item.gender ? <span className="text-[11px] text-muted">{item.gender === "male" ? "男" : item.gender === "female" ? "女" : item.gender}</span> : null}
                              {item.birth_date ? <span className="text-[11px] text-muted">{ageOf(item.birth_date)}</span> : null}
                            </span>
                            <span className="mt-0.5 block truncate text-[11px] text-muted">
                              {item.appointment_date || "未安排"} · {item.department || "未填科室"} · {item.doctor_name || "未指定医生"}
                            </span>
                          </span>
                          <span className={item.payment_status === "paid" ? "badge bg-emerald-50 text-emerald-700" : "badge bg-amber-50 text-amber-700"}>
                            {item.payment_status === "paid" ? "已缴费" : "待缴费"}
                          </span>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </section>

            <section className="card p-5">
              {!selected ? (
                <StateBlock kind="empty" title="选择左侧患者" description="查看患者 360：过敏、既往、用药、报告与随访" />
              ) : detailLoading ? (
                <div className="space-y-3">
                  <div className="h-16 animate-pulse rounded-2xl bg-canvas" />
                  <div className="h-24 animate-pulse rounded-2xl bg-canvas" />
                </div>
              ) : detailError ? (
                <StateBlock kind="error" title="读取患者资料失败" onRetry={() => selected && openPatient(selected)} />
              ) : detail ? (
                <div className="space-y-4">
                  <div className="flex flex-wrap items-center gap-3">
                    <h2 className="text-[18px] font-semibold text-ink">{detail.member.name}</h2>
                    <span className="badge">{detail.member.relation === "self" ? "本人" : detail.member.relation}</span>
                    {detail.member.gender ? <span className="text-[12px] text-muted">{detail.member.gender === "male" ? "男" : detail.member.gender === "female" ? "女" : detail.member.gender}</span> : null}
                    {detail.member.birth_date ? <span className="text-[12px] text-muted">{ageOf(detail.member.birth_date)}</span> : null}
                  </div>

                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="rounded-2xl bg-canvas p-3">
                      <p className="flex items-center gap-1.5 text-[12px] font-medium text-muted"><ShieldAlert size={13} /> 过敏史</p>
                      <p className="mt-1 text-[13px] text-ink">{detail.member.allergies || "未记录"}</p>
                    </div>
                    <div className="rounded-2xl bg-canvas p-3">
                      <p className="flex items-center gap-1.5 text-[12px] font-medium text-muted"><Activity size={13} /> 既往史</p>
                      <p className="mt-1 text-[13px] text-ink">{detail.member.conditions || "未记录"}</p>
                    </div>
                  </div>

                  {detail.previsit ? (
                    <div className="rounded-2xl border border-primary/20 bg-primary-soft/40 p-3">
                      <p className="flex items-center gap-1.5 text-[12px] font-medium text-primary">
                        <ClipboardList size={13} /> 就诊准备（患者提交）
                      </p>
                      <p className="mt-1 text-[13px] font-medium text-ink">{detail.previsit.chief_complaint}</p>
                      <div className="mt-1 space-y-0.5 text-[11px] leading-5 text-muted">
                        {detail.previsit.duration ? <p>持续时间：{detail.previsit.duration}</p> : null}
                        {detail.previsit.medications ? <p>在用药物：{detail.previsit.medications}</p> : null}
                        {detail.previsit.allergies ? <p>过敏史：{detail.previsit.allergies}</p> : null}
                        {detail.previsit.history ? <p>既往史：{detail.previsit.history}</p> : null}
                        {detail.previsit.questions ? <p>想问：{detail.previsit.questions}</p> : null}
                      </div>
                    </div>
                  ) : null}

                  <div>
                    <h3 className="flex items-center gap-1.5 text-[13px] font-medium text-ink"><FileText size={14} /> 报告（{detail.reports.length}）</h3>
                    <div className="mt-2 space-y-2">
                      {detail.reports.length === 0 ? <p className="rounded-2xl bg-canvas p-3 text-[12px] text-muted">未记录</p> : detail.reports.map((item) => (
                        <div key={item.id} className="rounded-2xl border border-line p-3">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="text-[13px] font-medium text-ink">{item.title}</span>
                            {item.critical_flags?.length ? <span className="badge bg-rose-50 text-rose-700">危急值</span> : null}
                            <span className="text-[11px] text-muted">{relativeTime(item.created_at)}</span>
                          </div>
                          <p className="mt-1 line-clamp-2 text-[12px] text-muted">{item.summary}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h3 className="flex items-center gap-1.5 text-[13px] font-medium text-ink"><Pill size={14} /> 用药（{detail.medications.length}）</h3>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {detail.medications.length === 0 ? <p className="rounded-2xl bg-canvas p-3 text-[12px] text-muted">未记录</p> : detail.medications.map((item) => (
                        <span key={item.id} className={item.active ? "chip" : "chip text-muted line-through"}>
                          {item.name}{item.dose ? " · " + item.dose : ""}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h3 className="flex items-center gap-1.5 text-[13px] font-medium text-ink"><CalendarCheck2 size={14} /> 随访（{detail.followups.length}）</h3>
                    <div className="mt-2 space-y-2">
                      {detail.followups.length === 0 ? <p className="rounded-2xl bg-canvas p-3 text-[12px] text-muted">未记录</p> : detail.followups.map((item) => (
                        <div key={item.id} className="flex items-center justify-between rounded-2xl bg-canvas px-3 py-2">
                          <span className="text-[13px] text-ink">{item.title}</span>
                          <span className="text-[11px] text-muted">{item.status === "done" ? "已完成" : item.due_date ? formatDate(item.due_date) : "待安排"}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h3 className="flex items-center gap-1.5 text-[13px] font-medium text-ink"><PenLine size={14} /> 病历草稿（{drafts.length}）</h3>
                    <textarea
                      className="input mt-2 min-h-[96px]"
                      value={draftText}
                      onChange={(event) => setDraftText(event.target.value)}
                      placeholder="主诉 / 现病史 / 查体 / 初步诊断（草稿，不写入 HIS）"
                    />
                    <div className="mt-2 flex flex-wrap items-center gap-2">
                      <button onClick={saveDraft} disabled={draftSaving || !draftText.trim()} className="btn-primary min-h-9 px-4 text-[12px] disabled:opacity-50">
                        {draftSaving ? "保存中…" : "保存草稿"}
                      </button>
                      <span className="text-[11px] text-muted">
                        {writeEnabled ? "签名将写入 HIS" : "签名未启用：需要医生 HIS 账号与签名透传"}
                      </span>
                    </div>
                    {draftError ? <p className="mt-2 text-[12px] text-rose-600">{draftError}</p> : null}
                    <div className="mt-3 space-y-2">
                      {drafts.length === 0 ? (
                        <p className="rounded-2xl bg-canvas p-3 text-[12px] text-muted">还没有草稿</p>
                      ) : (
                        drafts.map((item) => (
                          <div key={item.id} className="rounded-2xl border border-line p-3">
                            <p className="whitespace-pre-wrap text-[12.5px] leading-6 text-ink">{item.content}</p>
                            <div className="mt-2 flex items-center gap-2">
                              <span className={item.status === "signed" ? "badge bg-emerald-50 text-emerald-700" : "badge"}>
                                {item.status === "signed" ? "已签名" : "草稿"}
                              </span>
                              {item.status === "draft" ? (
                                <button
                                  onClick={() => signDraft(item)}
                                  disabled={!writeEnabled || signingId === item.id}
                                  title={writeEnabled ? "" : "HIS 写入未启用"}
                                  className="ml-auto text-[12px] font-medium text-primary disabled:text-muted"
                                >
                                  {signingId === item.id ? "签名中…" : "签名"}
                                </button>
                              ) : (
                                <span className="ml-auto text-[11px] text-muted">
                                  {item.signed_by || "医生"} · {item.signed_at ? relativeTime(item.signed_at) : ""}
                                </span>
                              )}
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </div>

                  <ClinicalWorkspace memberId={detail.member.id} writeEnabled={writeEnabled} />

                  <p className="border-t border-line pt-3 text-[11px] leading-5 text-muted">
                    写操作受门禁控制：病历与医嘱草稿可保存；签名需要医生 HIS 账号透传（当前 {writeEnabled ? "已启用" : "未启用"}）。
                  </p>
                </div>
              ) : null}
            </section>
          </div>
        )}
      </main>
    </div>
  );
}
