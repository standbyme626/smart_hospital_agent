"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  CalendarClock,
  CheckCircle2,
  ClipboardList,
  CreditCard,
  Loader2,
  MapPin,
  Pill,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Stethoscope,
  XCircle,
} from "lucide-react";
import clsx from "clsx";
import { AppShell } from "@/components/AppShell";
import { OfflineBanner, SourceBadge } from "@/components/ui";
import { useApp } from "@/lib/app-context";
import {
  ApiError,
  cancelAppointment,
  createAppointment,
  getAppointment,
  getAppointmentGuide,
  getAppointmentPrescriptions,
  getAppointments,
  getBookingCatalog,
  requestAppointmentPayment,
} from "@/lib/api";
import type {
  Appointment,
  AppointmentCancelResult,
  AppointmentCreateResult,
  AppointmentDetail,
  AppointmentGuide,
  BookingCatalog,
  PaymentTask,
  Prescription,
} from "@/lib/types";

const STEPS = ["选科室/医生", "选择挂号服务", "提交挂号（报价）", "待医院收费", "已挂号/完成"];

function todayIso(): string {
  const now = new Date();
  const local = new Date(now.getTime() - now.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 10);
}

function appointmentId(appointment: Appointment): string {
  return appointment.appointment_id || appointment.id;
}

function isPaidAppointment(appointment?: Appointment | null): boolean {
  return Boolean(appointment && appointment.payment_status === "paid");
}

function isRegisteredAppointment(appointment?: Appointment | null): boolean {
  if (!appointment) return false;
  if (["cancelled", "refunded"].includes(appointment.status)) return false;
  return (
    appointment.payment_status === "paid" ||
    ["registered", "completed"].includes(appointment.status)
  );
}

function isTerminalAppointment(appointment?: Appointment | null): boolean {
  return Boolean(appointment && ["cancelled", "refunded"].includes(appointment.status));
}

function statusLabel(appointment?: Appointment | null): string {
  if (!appointment) return "—";
  if (appointment.status === "cancelled") return "已取消";
  if (appointment.status === "refunded") return "已退费";
  if (isRegisteredAppointment(appointment)) return "已挂号";
  if (appointment.status === "awaiting_payment") return "待医院收费";
  return "待支付";
}

function statusClasses(appointment?: Appointment | null): string {
  if (!appointment) return "bg-canvas text-muted";
  if (isTerminalAppointment(appointment)) return "bg-canvas text-muted";
  if (isRegisteredAppointment(appointment)) return "bg-emerald-50 text-emerald-700";
  if (appointment.status === "awaiting_payment") return "bg-sky-50 text-sky-700";
  return "bg-amber-50 text-amber-700";
}

function formatTime(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

type PrescriptionState = "idle" | "loading" | "empty" | "ready" | "error";

export default function ServicesPage() {
  const { activeMember, authReady, showToast } = useApp();
  const memberId = activeMember?.id || null;

  const [today, setToday] = useState("");
  const [date, setDate] = useState("");
  const [catalog, setCatalog] = useState<BookingCatalog | null>(null);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [catalogAttempt, setCatalogAttempt] = useState(0);
  const [departmentId, setDepartmentId] = useState("");
  const [doctorId, setDoctorId] = useState("");
  const [serviceId, setServiceId] = useState("");

  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [ordersLoading, setOrdersLoading] = useState(false);
  const [ordersError, setOrdersError] = useState<string | null>(null);

  const [openId, setOpenId] = useState<string | null>(null);
  const [detail, setDetail] = useState<AppointmentDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [detailAttempt, setDetailAttempt] = useState(0);
  const [created, setCreated] = useState<AppointmentCreateResult | null>(null);

  const [task, setTask] = useState<PaymentTask | null>(null);
  const [taskError, setTaskError] = useState<string | null>(null);
  const [cancelInfo, setCancelInfo] = useState<AppointmentCancelResult | null>(null);
  const [createError, setCreateError] = useState<string | null>(null);
  const [missingIdCard, setMissingIdCard] = useState(false);
  const [polling, setPolling] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  const [guide, setGuide] = useState<AppointmentGuide | null>(null);
  const [guideError, setGuideError] = useState<string | null>(null);
  const [guideAttempt, setGuideAttempt] = useState(0);
  const [prescriptions, setPrescriptions] = useState<Prescription[] | null>(null);
  const [prescriptionState, setPrescriptionState] = useState<PrescriptionState>("idle");

  const baselineRef = useRef<string | null>(null);
  const submitKeyRef = useRef<string | null>(null);

  useEffect(() => {
    const value = todayIso();
    setToday(value);
    setDate((prev) => prev || value);
  }, []);

  const departments = catalog?.departments ?? [];
  const doctors = useMemo(
    () => (catalog?.doctors ?? []).filter((item) => !departmentId || item.org_id === departmentId),
    [catalog, departmentId],
  );
  const services = useMemo(
    () => (catalog?.services ?? []).filter((item) => !departmentId || item.org_id === departmentId),
    [catalog, departmentId],
  );

  useEffect(() => {
    if (!authReady || !date) return;
    let cancelled = false;
    (async () => {
      setCatalogLoading(true);
      setCatalogError(null);
      try {
        const data = await getBookingCatalog({ date });
        if (cancelled) return;
        setCatalog(data);
        setDepartmentId((prev) =>
          data.departments.some((item) => item.org_id === prev)
            ? prev
            : data.departments[0]?.org_id || "",
        );
      } catch (err) {
        if (cancelled) return;
        setCatalog(null);
        setCatalogError(err instanceof Error ? err.message : "医院目录加载失败");
      } finally {
        if (!cancelled) setCatalogLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [authReady, date, catalogAttempt]);

  const loadAppointments = useCallback(async (): Promise<Appointment[] | null> => {
    setOrdersLoading(true);
    setOrdersError(null);
    try {
      const items = await getAppointments(memberId || undefined);
      setAppointments(items);
      return items;
    } catch (err) {
      setOrdersError(err instanceof Error ? err.message : "挂号记录加载失败");
      return null;
    } finally {
      setOrdersLoading(false);
    }
  }, [memberId]);

  function openAppointment(id: string, syncUrl = true) {
    if (id !== openId) {
      setDetail((prev) => (prev && appointmentId(prev.appointment) === id ? prev : null));
      setDetailError(null);
      setCreated(null);
      setTask(null);
      setTaskError(null);
      setCancelInfo(null);
      setCreateError(null);
      setMissingIdCard(false);
      setPolling(false);
      baselineRef.current = null;
    }
    setOpenId(id);
    if (syncUrl && typeof window !== "undefined") {
      window.history.replaceState(null, "", `/services/?appointment=${encodeURIComponent(id)}`);
    }
  }

  useEffect(() => {
    if (!authReady) return;
    let cancelled = false;
    (async () => {
      setOpenId(null);
      setDetail(null);
      setDetailError(null);
      setCreated(null);
      setTask(null);
      setTaskError(null);
      setCancelInfo(null);
      setCreateError(null);
      setMissingIdCard(false);
      setPolling(false);
      baselineRef.current = null;
      const items = await loadAppointments();
      if (cancelled || !items) return;
      const deepLink = new URLSearchParams(window.location.search).get("appointment");
      if (deepLink) {
        openAppointment(deepLink, false);
        return;
      }
      const restorable = items.find((item) => !isTerminalAppointment(item));
      if (restorable) openAppointment(appointmentId(restorable), false);
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authReady, memberId]);

  useEffect(() => {
    if (!openId) {
      setDetail(null);
      setDetailError(null);
      setDetailLoading(false);
      return;
    }
    let cancelled = false;
    (async () => {
      setDetailLoading(true);
      setDetailError(null);
      try {
        const data = await getAppointment(openId);
        if (!cancelled) setDetail(data);
      } catch (err) {
        if (!cancelled) {
          setDetailError(err instanceof Error ? err.message : "挂号详情加载失败");
        }
      } finally {
        if (!cancelled) setDetailLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [openId, detailAttempt]);

  const activeAppointment = detail?.appointment ?? created?.appointment ?? null;
  const paid = isPaidAppointment(activeAppointment) || Boolean(detail && detail.payment.payment_status === "paid");
  const registered =
    isRegisteredAppointment(activeAppointment) ||
    Boolean(
      detail &&
        !isTerminalAppointment(activeAppointment) &&
        ["registered", "completed"].includes(detail.visit.state),
    );
  const terminal = isTerminalAppointment(activeAppointment);
  const stale = Boolean(
    activeAppointment?.stale || detail?.visit.stale || detail?.payment.stale,
  );
  const step = registered
    ? 4
    : activeAppointment
      ? activeAppointment.status === "awaiting_payment" || task?.status === "requested"
        ? 3
        : 2
      : serviceId
        ? 2
        : doctorId
          ? 1
          : 0;

  useEffect(() => {
    if (!openId || !registered) {
      setGuide(null);
      setGuideError(null);
      setPrescriptions(null);
      setPrescriptionState("idle");
      return;
    }
    let cancelled = false;
    (async () => {
      setGuideError(null);
      try {
        const data = await getAppointmentGuide(openId);
        if (!cancelled) setGuide(data.guide);
      } catch (err) {
        if (!cancelled) {
          setGuide(null);
          setGuideError(err instanceof Error ? err.message : "就诊指引加载失败");
        }
      }
      if (cancelled) return;
      setPrescriptionState("loading");
      try {
        const data = await getAppointmentPrescriptions(openId);
        if (cancelled) return;
        setPrescriptions(data.prescriptions);
        setPrescriptionState("ready");
      } catch (err) {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 409) {
          setPrescriptions([]);
          setPrescriptionState("empty");
        } else {
          setPrescriptions(null);
          setPrescriptionState("error");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [openId, registered, guideAttempt]);

  useEffect(() => {
    if (!polling || !openId) return;
    let cancelled = false;
    const id = openId;
    const tick = async () => {
      if (cancelled || document.hidden) return;
      try {
        const data = await getAppointment(id);
        if (cancelled) return;
        setDetail(data);
        const paymentStatus = data.payment?.payment_status || data.appointment?.payment_status;
        const state = data.visit?.state;
        if (paymentStatus === "paid") {
          setPolling(false);
          void loadAppointments();
          return;
        }
        if (data.visit?.stale) return;
        if (!baselineRef.current) {
          if (state) baselineRef.current = state;
          return;
        }
        if (state && state !== "unknown" && state !== baselineRef.current) {
          setPolling(false);
          void loadAppointments();
        }
      } catch {
        // transient HIS errors: keep polling until paid or the user leaves
      }
    };
    const timer = window.setInterval(tick, 3000);
    const onVisible = () => {
      if (!document.hidden) void tick();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [polling, openId, loadAppointments]);

  useEffect(() => {
    if (!openId || polling) return;
    const onVisible = () => {
      if (document.hidden) return;
      setDetailAttempt((value) => value + 1);
      void loadAppointments();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => document.removeEventListener("visibilitychange", onVisible);
  }, [openId, polling, loadAppointments]);

  function changeDepartment(value: string) {
    setDepartmentId(value);
    setDoctorId("");
    setServiceId("");
    setCreateError(null);
    setMissingIdCard(false);
  }

  function changeDate(value: string) {
    if (!value) return;
    setDate(value);
    setDoctorId("");
    setServiceId("");
    setCreateError(null);
    setMissingIdCard(false);
  }

  async function submitBooking() {
    const department = departments.find((item) => item.org_id === departmentId);
    const doctor = doctors.find((item) => item.practitioner_id === doctorId);
    const service = services.find((item) => item.service_id === serviceId);
    if (!department || !doctor || !service) {
      showToast("请先选择科室、医生和挂号服务");
      return;
    }
    if (!date) {
      showToast("请选择就诊日期");
      return;
    }
    setBusy("create");
    setCreateError(null);
    setMissingIdCard(false);
    setTask(null);
    setTaskError(null);
    setCancelInfo(null);
    submitKeyRef.current = [
      "appointment",
      activeMember?.id || "self",
      department.org_id,
      doctor.practitioner_id,
      service.service_id,
      date,
    ].join(":");
    try {
      const result = await createAppointment({
        member_id: activeMember?.id,
        org_id: department.org_id,
        practitioner_id: doctor.practitioner_id,
        service_id: service.service_id,
        date,
        idempotency_key: submitKeyRef.current,
      });
      submitKeyRef.current = null;
      setCreated(result);
      setDetail(null);
      setDetailError(null);
      setOpenId(appointmentId(result.appointment));
      setPolling(false);
      baselineRef.current = null;
      window.history.replaceState(
        null,
        "",
        `/services/?appointment=${encodeURIComponent(appointmentId(result.appointment))}`,
      );
      showToast("已生成挂号报价");
      void loadAppointments();
    } catch (err) {
      if (err instanceof ApiError && err.status === 409 && err.message.includes("身份证")) {
        setMissingIdCard(true);
      } else {
        setCreateError(err instanceof Error ? err.message : "提交挂号失败");
      }
    } finally {
      setBusy(null);
    }
  }

  async function doRequestPayment() {
    if (!activeAppointment) return;
    const id = appointmentId(activeAppointment);
    setBusy("payment");
    setTaskError(null);
    baselineRef.current = detail?.visit?.state || null;
    try {
      const result = await requestAppointmentPayment(id);
      setTask(result.task);
      setDetail((prev) =>
        prev ? { ...prev, appointment: result.appointment } : { appointment: result.appointment, visit: { state: result.appointment.status }, payment: { payment_status: result.appointment.payment_status } },
      );
      if (result.task.status !== "paid") setPolling(true);
      showToast("已生成收费任务，支付后状态会自动同步（可能延迟数十秒）");
      void loadAppointments();
    } catch (err) {
      setTaskError(err instanceof Error ? err.message : "申请缴费失败");
    } finally {
      setBusy(null);
    }
  }

  async function doCancel() {
    if (!activeAppointment) return;
    const id = appointmentId(activeAppointment);
    const paidAlready = paid || registered;
    const confirmed = window.confirm(
      paidAlready
        ? "该挂号已支付，退号/退费需医院处理。确认提交退号申请？"
        : "确认取消该挂号？",
    );
    if (!confirmed) return;
    setBusy("cancel");
    try {
      const result = await cancelAppointment(id, { reason: "user_cancelled", refund: paidAlready });
      setCancelInfo(result.cancel);
      setTask(null);
      setTaskError(null);
      setPolling(false);
      const refreshed = await getAppointment(id).catch(() => null);
      if (refreshed) {
        setDetail(refreshed);
      } else {
        setDetail((prev) =>
          prev
            ? { ...prev, appointment: result.appointment }
            : {
                appointment: result.appointment,
                visit: { state: result.appointment.status },
                payment: { payment_status: result.appointment.payment_status },
              },
        );
      }
      showToast(
        paidAlready
          ? "退号/退费请求已提交，以医院处理结果为准"
          : "取消请求已提交，医院状态会自动同步",
      );
      void loadAppointments();
    } catch (err) {
      showToast(err instanceof Error ? err.message : "取消失败");
    } finally {
      setBusy(null);
    }
  }

  function rebook() {
    setOpenId(null);
    setDetail(null);
    setDetailError(null);
    setCreated(null);
    setTask(null);
    setTaskError(null);
    setCancelInfo(null);
    setCreateError(null);
    setMissingIdCard(false);
    setPolling(false);
    baselineRef.current = null;
    window.history.replaceState(null, "", "/services/");
  }

  return (
    <AppShell
      title="挂号服务"
      subtitle="HIS 报价 + 医院收费；支付以医院收费窗口为准"
      actions={<SourceBadge live />}
    >
      <div className="mx-auto max-w-4xl space-y-4">
        <OfflineBanner />
        <div className="flex flex-wrap items-center gap-2">
          {STEPS.map((label, index) => (
            <div key={label} className="flex items-center gap-2">
              <span
                className={clsx(
                  "flex h-6 w-6 items-center justify-center rounded-full text-[11px]",
                  index < step
                    ? "bg-emerald-100 text-emerald-600"
                    : index === step
                      ? "bg-primary text-white"
                      : "bg-canvas text-muted",
                )}
              >
                {index < step ? <CheckCircle2 size={13} /> : index + 1}
              </span>
              <span className={clsx("text-[12px]", index <= step ? "text-ink" : "text-muted")}>
                {label}
              </span>
              {index < STEPS.length - 1 ? <span className="hidden h-px w-4 bg-line sm:block" /> : null}
            </div>
          ))}
        </div>

        <div className="flex items-start gap-2 rounded-2xl border border-line bg-card p-3 text-[11px] leading-5 text-muted">
          <ShieldCheck size={14} className="mt-0.5 shrink-0 text-primary" />
          当前为 HIS 适配器 mock/试点模式，支付以医院收费窗口为准；状态会自动同步（可能延迟数十秒），不承诺实时。
        </div>

        <section className="card p-5">
          <div className="flex items-center justify-between">
            <p className="section-title flex items-center gap-1.5">
              <CalendarClock size={15} className="text-primary" />
              医院目录
            </p>
            {catalog?.source ? (
              <span className="text-[11px] text-muted">
                来源 {catalog.source}
                {catalog.stale ? " · 缓存数据，可能延迟" : ""}
              </span>
            ) : null}
          </div>

          {catalogError ? (
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-2xl bg-rose-50 p-3 text-[12px] text-rose-700">
              <span className="flex items-center gap-1.5">
                <AlertCircle size={14} />医院系统暂不可用：{catalogError}
              </span>
              <button onClick={() => setCatalogAttempt((value) => value + 1)} className="btn-ghost">
                <RefreshCw size={14} />重试
              </button>
            </div>
          ) : catalogLoading && !catalog ? (
            <div className="mt-3 flex items-center gap-2 text-[12px] text-muted">
              <Loader2 size={14} className="animate-spin" />加载医院目录…
            </div>
          ) : (
            <>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <label className="block">
                  <span className="label">科室</span>
                  <select
                    className="input"
                    value={departmentId}
                    onChange={(event) => changeDepartment(event.target.value)}
                  >
                    {departments.map((item) => (
                      <option key={item.org_id} value={item.org_id}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="block">
                  <span className="label">就诊日期</span>
                  <input
                    className="input"
                    type="date"
                    value={date}
                    min={today || undefined}
                    onChange={(event) => changeDate(event.target.value)}
                  />
                </label>
              </div>

              {departments.length === 0 ? (
                <p className="mt-3 rounded-2xl bg-canvas p-4 text-[12px] text-muted">
                  医院暂未返回可挂号科室，请稍后重试。
                </p>
              ) : null}

              <div className="mt-5">
                <p className="section-title flex items-center gap-1.5">
                  <Stethoscope size={15} className="text-primary" />医生
                </p>
                <div className="mt-2 grid gap-2 sm:grid-cols-3">
                  {doctors.map((item) => (
                    <button
                      key={item.practitioner_id}
                      onClick={() => {
                        setDoctorId(item.practitioner_id);
                        setServiceId("");
                        setCreateError(null);
                        setMissingIdCard(false);
                      }}
                      className={clsx(
                        "rounded-2xl border p-3 text-left transition",
                        doctorId === item.practitioner_id
                          ? "border-primary bg-primary-soft"
                          : "border-line bg-white hover:border-primary/30",
                      )}
                    >
                      <span className="text-[13px] font-semibold text-ink">{item.name}</span>
                      <p className="mt-0.5 text-[11px] text-muted">预约挂号</p>
                    </button>
                  ))}
                </div>
                {doctors.length === 0 ? (
                  <p className="mt-2 text-[11px] text-muted">该科室暂未返回医生。</p>
                ) : null}
              </div>

              <div className="mt-5">
                <p className="section-title flex items-center gap-1.5">
                  <ClipboardList size={15} className="text-primary" />挂号服务
                </p>
                <div className="mt-2 grid gap-2 sm:grid-cols-2">
                  {services.map((item) => (
                    <button
                      key={item.service_id}
                      onClick={() => {
                        setServiceId(item.service_id);
                        setCreateError(null);
                        setMissingIdCard(false);
                      }}
                      className={clsx(
                        "flex items-center justify-between rounded-2xl border p-3 text-left transition",
                        serviceId === item.service_id
                          ? "border-primary bg-primary-soft"
                          : "border-line bg-white hover:border-primary/30",
                      )}
                    >
                      <span>
                        <span className="block text-[13px] font-semibold text-ink">{item.name}</span>
                        <span className="mt-0.5 block text-[11px] text-muted">{item.service_id}</span>
                      </span>
                      <span className="text-[13px] font-semibold text-primary">
                        ¥{Number(item.price || 0).toFixed(2)}
                      </span>
                    </button>
                  ))}
                </div>
                {services.length === 0 ? (
                  <p className="mt-2 text-[11px] text-muted">该科室暂未返回可挂号服务。</p>
                ) : null}
              </div>

              <div className="mt-5 flex flex-wrap items-center gap-2">
                <button
                  onClick={submitBooking}
                  disabled={busy !== null || !departmentId || !doctorId || !serviceId || !date}
                  className="btn-primary"
                >
                  {busy === "create" ? (
                    <Loader2 size={15} className="animate-spin" />
                  ) : (
                    <CalendarClock size={15} />
                  )}
                  提交挂号（获取报价）
                </button>
                <span className="text-[11px] text-muted">
                  就诊人：{activeMember ? `${activeMember.name}（${activeMember.relation}）` : "本人档案"}
                </span>
              </div>

              {missingIdCard ? (
                <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-2xl bg-amber-50 p-3 text-[12px] text-amber-800">
                  <span>请先在家庭档案补充就诊人身份证号，挂号需要实名信息。</span>
                  <Link href="/profile/" className="btn-primary">
                    去补充
                  </Link>
                </div>
              ) : null}

              {createError ? (
                <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-2xl bg-rose-50 p-3 text-[12px] text-rose-700">
                  <span className="flex items-center gap-1.5">
                    <AlertCircle size={14} />提交失败：{createError}
                  </span>
                  <button onClick={submitBooking} disabled={busy !== null} className="btn-ghost">
                    <RefreshCw size={14} />重试
                  </button>
                </div>
              ) : null}
            </>
          )}
        </section>

        {detailError ? (
          <section className="card flex flex-wrap items-center justify-between gap-2 p-5">
            <p className="flex items-center gap-1.5 text-[12px] text-rose-700">
              <AlertCircle size={14} />挂号详情加载失败：{detailError}
            </p>
            <button onClick={() => setDetailAttempt((value) => value + 1)} className="btn-ghost">
              <RefreshCw size={14} />重试
            </button>
          </section>
        ) : null}

        {detailLoading && !activeAppointment ? (
          <div className="flex items-center gap-2 text-[12px] text-muted">
            <Loader2 size={14} className="animate-spin" />加载挂号详情…
          </div>
        ) : null}

        {activeAppointment ? (
          <section className="card p-5">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div>
                <p className="text-[11px] text-muted">挂号单</p>
                <p className="mt-1 break-all text-[15px] font-semibold text-ink">
                  {appointmentId(activeAppointment)}
                </p>
              </div>
              <span className={clsx("badge", statusClasses(activeAppointment))}>
                {statusLabel(activeAppointment)}
              </span>
            </div>

            <div className="mt-3 grid gap-2 text-[12px] text-muted sm:grid-cols-2">
              <p>
                金额：<span className="font-semibold text-ink">¥{Number(activeAppointment.amount || 0).toFixed(2)}</span>
              </p>
              <p>
                报价单号：<span className="break-all text-ink">{activeAppointment.quote_id || "—"}</span>
              </p>
              <p>
                科室/医生：
                <span className="text-ink">
                  {activeAppointment.department || "—"}
                  {activeAppointment.doctor_name ? ` · ${activeAppointment.doctor_name}` : ""}
                </span>
              </p>
              <p>
                就诊日期：<span className="text-ink">{activeAppointment.appointment_date || date || "—"}</span>
              </p>
              {detail?.visit.encounter_id ? (
                <p>
                  就诊号：<span className="break-all text-ink">{detail.visit.encounter_id}</span>
                </p>
              ) : null}
              <p>
                就诊人：<span className="text-ink">{activeMember?.name || "本人档案"}</span>
              </p>
              <p>
                创建时间：<span className="text-ink">{formatTime(activeAppointment.created_at)}</span>
              </p>
            </div>

            {created?.instructions ? (
              <p className="mt-3 rounded-2xl bg-canvas p-3 text-[12px] text-ink">
                {created.instructions}
              </p>
            ) : null}
            {created?.payment_mode ? (
              <p className="mt-2 text-[11px] text-muted">
                支付模式：{created.payment_mode}（试点，以医院收费窗口为准）
              </p>
            ) : null}

            {!paid && !terminal ? (
              <p className="mt-3 text-[12px] text-amber-700">
                当前状态：
                {activeAppointment.status === "awaiting_payment" ? "待医院收费" : "待支付/待医院登记"}。
                请到收费窗口/自助机支付，支付后状态自动同步（可能延迟数十秒）。
              </p>
            ) : null}

            {task ? (
              <div className="mt-3 rounded-2xl bg-amber-50 p-3 text-[12px] text-amber-900">
                <p className="font-medium">
                  收费任务已创建{task.task_id ? `（${task.task_id}）` : ""}
                </p>
                <p className="mt-1">{task.instructions || "请到收费窗口或自助机完成支付。"}</p>
                {task.sla_due_at ? (
                  <p className="mt-1 text-[11px]">医院处理时限：{formatTime(task.sla_due_at)}</p>
                ) : null}
                {polling ? (
                  <p className="mt-1 flex items-center gap-1.5 text-[11px]">
                    <Loader2 size={12} className="animate-spin" />每 3 秒自动同步一次状态…
                  </p>
                ) : null}
              </div>
            ) : null}

            {taskError ? (
              <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-2xl bg-rose-50 p-3 text-[12px] text-rose-700">
                <span className="flex items-center gap-1.5">
                  <AlertCircle size={14} />申请缴费失败：{taskError}
                </span>
                <button onClick={doRequestPayment} disabled={busy !== null} className="btn-ghost">
                  <RefreshCw size={14} />重试
                </button>
              </div>
            ) : null}

            {cancelInfo ? (
              <p className="mt-3 rounded-2xl bg-canvas p-3 text-[12px] text-ink">
                医院返回状态：{cancelInfo.status}
                {cancelInfo.encounter_id ? ` · 就诊号 ${cancelInfo.encounter_id}` : ""}
                {cancelInfo.refund_id ? ` · 退费单 ${cancelInfo.refund_id}` : ""}
                {paid ? "（退费到账时间以医院为准）" : ""}
              </p>
            ) : null}

            {stale ? (
              <p className="mt-3 flex items-center gap-1.5 rounded-2xl bg-amber-50 p-3 text-[11px] text-amber-800">
                <AlertCircle size={13} />医院系统暂不可用，当前显示上次同步结果。
              </p>
            ) : null}

            <div className="mt-4 flex flex-wrap gap-2">
              {!paid && !terminal ? (
                <>
                  <button onClick={doRequestPayment} disabled={busy !== null} className="btn-primary">
                    {busy === "payment" ? (
                      <Loader2 size={15} className="animate-spin" />
                    ) : (
                      <CreditCard size={15} />
                    )}
                    去缴费 / 申请支付
                  </button>
                  <button
                    onClick={doCancel}
                    disabled={busy !== null}
                    className="btn-ghost text-rose-600"
                  >
                    <XCircle size={15} />取消挂号
                  </button>
                </>
              ) : null}
              {paid && !terminal ? (
                <button onClick={doCancel} disabled={busy !== null} className="btn-ghost text-rose-600">
                  <XCircle size={15} />退号/退费（医院处理）
                </button>
              ) : null}
              {registered || terminal ? (
                <button onClick={rebook} className="btn-ghost">
                  <RotateCcw size={15} />再约一次
                </button>
              ) : null}
              {!paid && !terminal ? (
                <button
                  onClick={() => setDetailAttempt((value) => value + 1)}
                  disabled={detailLoading}
                  className="btn-ghost"
                >
                  <RefreshCw size={15} className={detailLoading ? "animate-spin" : ""} />
                  刷新状态
                </button>
              ) : null}
            </div>
          </section>
        ) : null}

        {guideError ? (
          <section className="card flex flex-wrap items-center justify-between gap-2 p-5">
            <p className="text-[12px] text-rose-700">就诊指引加载失败：{guideError}</p>
            <button onClick={() => setGuideAttempt((value) => value + 1)} className="btn-ghost">
              <RotateCcw size={15} />重试
            </button>
          </section>
        ) : null}

        {registered && guide ? (
          <section className="card p-5">
            <p className="section-title flex items-center gap-1.5">
              <MapPin size={15} className="text-primary" />就诊指引
            </p>
            <p className="mt-2 text-[13px] font-medium text-ink">
              {guide.hospital}
              {guide.floor ? ` · ${guide.floor}` : ""}
            </p>
            {guide.address ? <p className="mt-1 text-[12px] text-muted">{guide.address}</p> : null}
            <p className="mt-2 text-[12px] text-ink">{guide.checkin}</p>
            {guide.items.length ? (
              <ul className="mt-2 list-disc space-y-1 pl-5 text-[12px] text-muted">
                {guide.items.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            ) : null}
            <p className="mt-2 text-[11px] text-muted">{guide.support}</p>
            {guide.stale ? (
              <p className="mt-2 text-[11px] text-amber-700">指引来自上次同步缓存，可能延迟。</p>
            ) : null}
          </section>
        ) : null}

        {registered ? (
          <section className="card p-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="section-title flex items-center gap-1.5">
                <Pill size={15} className="text-primary" />处方
              </p>
              {prescriptionState === "error" ? (
                <button onClick={() => setGuideAttempt((value) => value + 1)} className="btn-ghost">
                  <RefreshCw size={14} />重试
                </button>
              ) : null}
            </div>
            {prescriptionState === "loading" ? (
              <p className="mt-2 flex items-center gap-1.5 text-[12px] text-muted">
                <Loader2 size={14} className="animate-spin" />加载处方…
              </p>
            ) : prescriptionState === "empty" ? (
              <p className="mt-2 text-[12px] text-muted">暂无处方（医院尚未生成就诊记录）。</p>
            ) : prescriptionState === "error" ? (
              <p className="mt-2 text-[12px] text-rose-700">处方加载失败，请稍后重试。</p>
            ) : prescriptions && prescriptions.length > 0 ? (
              <ul className="mt-2 space-y-2">
                {prescriptions.map((item) => (
                  <li key={item.his_prescription_no} className="rounded-2xl bg-canvas p-3 text-[12px]">
                    <p className="font-medium text-ink">{item.his_prescription_no}</p>
                    <p className="mt-0.5 text-[11px] text-muted">
                      {item.status || "状态未知"} · {formatTime(item.issued_at)}
                    </p>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-2 text-[12px] text-muted">暂无处方。</p>
            )}
          </section>
        ) : null}

        <section className="card p-5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="section-title flex items-center gap-1.5">
              <ClipboardList size={15} className="text-primary" />我的挂号
            </p>
            <button
              onClick={() => void loadAppointments()}
              disabled={ordersLoading}
              className="btn-ghost"
            >
              <RefreshCw size={14} className={ordersLoading ? "animate-spin" : ""} />刷新
            </button>
          </div>

          {ordersError ? (
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-2xl bg-rose-50 p-3 text-[12px] text-rose-700">
              <span className="flex items-center gap-1.5">
                <AlertCircle size={14} />挂号记录加载失败：{ordersError}
              </span>
              <button onClick={() => void loadAppointments()} className="btn-ghost">
                <RefreshCw size={14} />重试
              </button>
            </div>
          ) : ordersLoading && appointments.length === 0 ? (
            <p className="mt-3 flex items-center gap-2 text-[12px] text-muted">
              <Loader2 size={14} className="animate-spin" />加载挂号记录…
            </p>
          ) : appointments.length === 0 ? (
            <p className="mt-3 rounded-2xl bg-canvas p-4 text-[12px] text-muted">
              暂无挂号记录。选择科室、医生和挂号服务后提交挂号。
            </p>
          ) : (
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              {appointments.map((item) => {
                const id = appointmentId(item);
                return (
                  <button
                    key={id}
                    onClick={() => openAppointment(id)}
                    className={clsx(
                      "rounded-2xl border p-3 text-left transition",
                      openId === id
                        ? "border-primary bg-primary-soft"
                        : "border-line bg-white hover:border-primary/30",
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[13px] font-semibold text-ink">
                        {item.department || "门诊"}
                        {item.doctor_name ? ` · ${item.doctor_name}` : ""}
                      </span>
                      <span className={clsx("badge", statusClasses(item))}>
                        {statusLabel(item)}
                      </span>
                    </div>
                    <p className="mt-1 text-[11px] text-muted">
                      {item.appointment_date || "日期待同步"} · ¥
                      {Number(item.amount || 0).toFixed(2)}
                    </p>
                    <p className="mt-0.5 break-all text-[10px] text-muted">{id}</p>
                  </button>
                );
              })}
            </div>
          )}
        </section>

        <div className="flex items-start gap-2 rounded-2xl border border-line bg-card p-3 text-[11px] leading-5 text-muted">
          <ShieldCheck size={14} className="mt-0.5 shrink-0 text-primary" />
          挂号由 HIS 生成报价，支付与退费以医院收费窗口/自助机为准；本页面只同步医院返回的状态，不代替医院完成收费。
        </div>
      </div>
    </AppShell>
  );
}
