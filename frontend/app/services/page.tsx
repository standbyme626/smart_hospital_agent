"use client";

import { useEffect, useMemo, useState } from "react";
import {
  CalendarClock,
  CheckCircle2,
  CreditCard,
  Loader2,
  Lock,
  MapPin,
  ShieldCheck,
  Star,
  Stethoscope,
} from "lucide-react";
import clsx from "clsx";
import { AppShell } from "@/components/AppShell";
import { useApp } from "@/lib/app-context";
import {
  createOrder,
  getDepartments,
  getDoctors,
  getOrder,
  getSchedules,
  getVisitGuide,
  lockSlot,
  requestPayment,
  track,
} from "@/lib/api";
import type { BookingOrder, Doctor, ReviewRequest, ScheduleSlot, VisitGuide } from "@/lib/types";

const STEPS = ["选号", "锁号", "订单", "支付审核", "完成"];

export default function ServicesPage() {
  const { activeMember, showToast } = useApp();
  const [departments, setDepartments] = useState<string[]>([]);
  const [department, setDepartment] = useState("心血管内科");
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [doctor, setDoctor] = useState<Doctor | null>(null);
  const [date, setDate] = useState("");
  const [slots, setSlots] = useState<ScheduleSlot[]>([]);
  const [selected, setSelected] = useState<ScheduleSlot | null>(null);
  const [lock, setLock] = useState<{ lock_id: string; slot_id: string; expires_in: number } | null>(null);
  const [order, setOrder] = useState<BookingOrder | null>(null);
  const [review, setReview] = useState<ReviewRequest | null>(null);
  const [guide, setGuide] = useState<VisitGuide | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const saved = window.sessionStorage.getItem("booking_department");
    if (saved) {
      setDepartment(saved);
      window.sessionStorage.removeItem("booking_department");
    }
    getDepartments()
      .then((items) => setDepartments(items))
      .catch(() => setDepartments([]));
  }, []);

  useEffect(() => {
    load();
    track("services_view", { department });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [department, date]);

  async function load() {
    setLoading(true);
    setError(null);
    setDoctor(null);
    setSelected(null);
    setLock(null);
    setOrder(null);
    setReview(null);
    setGuide(null);
    try {
      const [doctorList, slotList] = await Promise.all([
        getDoctors(department),
        getSchedules(department, date || undefined),
      ]);
      setDoctors(doctorList);
      setSlots(slotList);
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }

  const grouped = useMemo(() => {
    const source = doctor ? slots.filter((slot) => slot.doctor === doctor.name) : slots;
    const map = new Map<string, ScheduleSlot[]>();
    for (const slot of source) {
      const list = map.get(slot.date) || [];
      list.push(slot);
      map.set(slot.date, list);
    }
    return Array.from(map.entries()).slice(0, 4);
  }, [slots, doctor]);

  const pendingReview = Boolean(review && ["pending", "processing"].includes(review.status));
  const paid = order?.payment_status === "paid";
  const step = paid ? 4 : pendingReview ? 3 : order ? 2 : lock ? 1 : 0;

  useEffect(() => {
    if (!order || !pendingReview) return;
    const orderId = order.order_id;
    const timer = window.setInterval(async () => {
      try {
        const result = await getOrder(orderId);
        setOrder(result.order);
        setReview(result.review);
      } catch {
        // transient errors while the admin approves are expected; keep polling
      }
    }, 3000);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [order?.order_id, pendingReview]);

  useEffect(() => {
    if (!order || order.payment_status !== "paid" || guide) return;
    getVisitGuide(order.order_id)
      .then(setGuide)
      .catch(() => setGuide(null));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [order?.order_id, order?.payment_status, guide]);

  async function doLock() {
    if (!selected) return;
    setLoading(true);
    try {
      setLock(await lockSlot(selected.slot_id, activeMember?.id));
      track("booking_lock", { slot_id: selected.slot_id });
    } catch (err) {
      showToast(err instanceof Error ? err.message : "锁号失败");
    } finally {
      setLoading(false);
    }
  }

  async function doOrder() {
    if (!lock) return;
    setLoading(true);
    try {
      setOrder(await createOrder(lock.lock_id, activeMember?.id));
      setReview(null);
      setGuide(null);
      track("booking_order", { lock_id: lock.lock_id });
    } catch (err) {
      showToast(err instanceof Error ? err.message : "创建订单失败");
    } finally {
      setLoading(false);
    }
  }

  async function doRequestPayment() {
    if (!order) return;
    setLoading(true);
    try {
      const result = await requestPayment(order.order_id);
      setReview(result.review);
      track("payment_request", { order_id: order.order_id });
    } catch (err) {
      showToast(err instanceof Error ? err.message : "申请支付失败");
    } finally {
      setLoading(false);
    }
  }

  const statusLabel = paid
    ? "已支付"
    : review?.status === "rejected"
      ? "已拒绝"
      : review?.status === "failed"
        ? "审核失败"
        : pendingReview
          ? "等待人工确认"
          : "待支付";

  return (
    <AppShell title="挂号服务" subtitle="选科室、医生与号源；支付申请由管理员人工审核">
      <div className="mx-auto max-w-4xl space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          {STEPS.map((label, index) => (
            <div key={label} className="flex items-center gap-2">
              <span className={clsx("flex h-6 w-6 items-center justify-center rounded-full text-[11px]", index < step ? "bg-emerald-100 text-emerald-600" : index === step ? "bg-primary text-white" : "bg-canvas text-muted")}>
                {index < step ? <CheckCircle2 size={13} /> : index + 1}
              </span>
              <span className={clsx("text-[12px]", index <= step ? "text-ink" : "text-muted")}>{label}</span>
              {index < STEPS.length - 1 ? <span className="hidden h-px w-4 bg-line sm:block" /> : null}
            </div>
          ))}
        </div>

        {error ? <div className="rounded-2xl bg-rose-50 p-3 text-[12px] text-rose-700">{error}</div> : null}

        <section className="card p-5">
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="block">
              <span className="label">科室</span>
              <select className="input" value={department} onChange={(event) => setDepartment(event.target.value)}>
                {departments.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </label>
            <label className="block">
              <span className="label">日期（可选）</span>
              <input className="input" type="date" value={date} onChange={(event) => setDate(event.target.value)} />
            </label>
          </div>

          <div className="mt-5">
            <p className="section-title flex items-center gap-1.5"><Stethoscope size={15} className="text-primary" />医生</p>
            <div className="mt-2 grid gap-2 sm:grid-cols-3">
              {doctors.map((item) => (
                <button
                  key={item.doctor_id}
                  onClick={() => { setDoctor(item); setSelected(null); }}
                  className={clsx(
                    "rounded-2xl border p-3 text-left transition",
                    doctor?.doctor_id === item.doctor_id ? "border-primary bg-primary-soft" : "border-line bg-white hover:border-primary/30",
                  )}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[13px] font-semibold text-ink">{item.name}</span>
                    <span className="flex items-center gap-0.5 text-[11px] text-amber-500"><Star size={11} fill="currentColor" />{item.rating}</span>
                  </div>
                  <p className="mt-0.5 text-[11px] text-muted">{item.title}</p>
                  <p className="mt-1 line-clamp-2 text-[11px] leading-5 text-muted">{item.intro}</p>
                  <p className="mt-1 text-[11px] text-primary">¥{item.fee}</p>
                </button>
              ))}
            </div>
            {doctor ? <button onClick={() => setDoctor(null)} className="mt-2 text-[11px] text-muted underline underline-offset-2">查看全部医生号源</button> : null}
          </div>
        </section>

        <section className="card p-5">
          <div className="flex items-center justify-between">
            <p className="section-title flex items-center gap-1.5"><CalendarClock size={15} className="text-primary" />可预约号源</p>
            <span className="text-[11px] text-muted">{doctor ? doctor.name : "全部医生"}</span>
          </div>
          {loading ? (
            <div className="mt-3 flex items-center gap-2 text-[12px] text-muted"><Loader2 size={14} className="animate-spin" />加载中…</div>
          ) : (
            <div className="mt-3 space-y-4">
              {grouped.map(([day, items]) => (
                <div key={day}>
                  <p className="mb-2 text-[11px] text-muted">{day}</p>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    {items.map((slot) => (
                      <button
                        key={slot.slot_id}
                        disabled={!slot.available}
                        onClick={() => { setSelected(slot); setLock(null); setOrder(null); setReview(null); setGuide(null); }}
                        className={clsx(
                          "rounded-2xl border px-3 py-2.5 text-left text-[12px] transition",
                          selected?.slot_id === slot.slot_id ? "border-primary bg-primary-soft text-primary" : "border-line bg-white text-ink hover:border-primary/30",
                          !slot.available ? "opacity-40" : "",
                        )}
                      >
                        <span className="block font-medium">{slot.time}</span>
                        <span className="mt-0.5 block text-[11px] text-muted">{slot.doctor} · ¥{slot.fee}</span>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
              {grouped.length === 0 ? <p className="rounded-2xl bg-canvas p-4 text-[12px] text-muted">暂无可预约号源，换个日期或科室试试。</p> : null}
            </div>
          )}
        </section>

        <section className="card p-5">
          <div className="flex flex-wrap gap-2">
            <button onClick={doLock} disabled={!selected || Boolean(lock) || loading} className="btn-ghost"><Lock size={15} />锁定号源</button>
            <button onClick={doOrder} disabled={!lock || Boolean(order) || loading} className="btn-ghost"><Stethoscope size={15} />创建订单</button>
            <button
              onClick={doRequestPayment}
              disabled={!order || pendingReview || paid || loading}
              className="btn-primary"
            >
              <CreditCard size={15} />{review?.status === "rejected" ? "重新申请支付" : "申请支付"}
            </button>
          </div>
          <div className="mt-3 rounded-2xl bg-canvas p-3 text-[11px] text-muted">
            <p className="font-medium text-ink">当前锁定</p>
            <p className="mt-1 break-all">{lock ? lock.lock_id : "尚未锁定号源"}</p>
            {activeMember ? <p className="mt-1">就诊人：{activeMember.name}（{activeMember.relation}）</p> : null}
          </div>
        </section>

        {order ? (
          <section className="card p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-[11px] text-muted">订单</p>
                <p className="mt-1 text-[15px] font-semibold text-ink">{order.order_id}</p>
              </div>
              <span className={clsx("badge", paid ? "bg-emerald-50 text-emerald-700" : review && ["rejected", "failed"].includes(review.status) ? "bg-rose-50 text-rose-700" : "bg-amber-50 text-amber-700")}>
                {statusLabel}
              </span>
            </div>
            <p className="mt-2 text-[12px] text-muted">{order.slot_id} · ¥{order.amount}</p>
            {pendingReview ? (
              <p className="mt-2 flex items-center gap-1.5 text-[11px] text-amber-700"><Loader2 size={12} className="animate-spin" />已提交支付申请，等待管理员人工确认（每 3 秒自动刷新）。</p>
            ) : null}
            {review?.status === "rejected" ? (
              <p className="mt-2 text-[11px] text-rose-700">申请被拒绝{review.decision_note ? `：${review.decision_note}` : "。"}你可以修改后重新申请。</p>
            ) : null}
            {review?.status === "failed" ? (
              <p className="mt-2 text-[11px] text-rose-700">支付调用失败，请重新申请或联系管理员。</p>
            ) : null}
          </section>
        ) : null}

        {guide ? (
          <section className="card p-5">
            <p className="section-title flex items-center gap-1.5"><MapPin size={15} className="text-primary" />就诊指引</p>
            <p className="mt-2 text-[13px] font-medium text-ink">{guide.hospital} · {guide.floor}</p>
            <p className="mt-1 text-[12px] text-muted">{guide.address}</p>
            <p className="mt-2 text-[12px] text-ink">{guide.checkin}</p>
            <ul className="mt-2 list-disc space-y-1 pl-5 text-[12px] text-muted">
              {guide.items.map((item) => <li key={item}>{item}</li>)}
            </ul>
            <p className="mt-2 text-[11px] text-muted">{guide.support}</p>
          </section>
        ) : null}

        <div className="flex items-start gap-2 rounded-2xl border border-line bg-card p-3 text-[11px] leading-5 text-muted">
          <ShieldCheck size={14} className="mt-0.5 shrink-0 text-primary" />
          支付申请提交后必须由管理员人工批准，批准前不会发起扣款。当前为 mock 医院 MCP，真实环境将替换为医院系统。
        </div>
      </div>
    </AppShell>
  );
}
