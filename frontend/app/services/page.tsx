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
  getSchedules,
  getVisitGuide,
  lockSlot,
  payOrder,
  track,
} from "@/lib/api";
import type { BookingOrder, Doctor, ScheduleSlot, VisitGuide } from "@/lib/types";

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
  const [guide, setGuide] = useState<VisitGuide | null>(null);
  const [confirmToken, setConfirmToken] = useState("confirm-demo");
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

  const step = guide ? 4 : order?.payment_status === "paid" ? 3 : order ? 2 : lock ? 1 : 0;

  async function doLock() {
    if (!selected) return;
    setLoading(true);
    try {
      setLock(await lockSlot(selected.slot_id, activeMember?.id || "P001"));
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
      setOrder(await createOrder(lock.lock_id, activeMember?.id || "P001"));
      track("booking_order", { lock_id: lock.lock_id });
    } catch (err) {
      showToast(err instanceof Error ? err.message : "创建订单失败");
    } finally {
      setLoading(false);
    }
  }

  async function doPay() {
    if (!order) return;
    setLoading(true);
    try {
      const paid = await payOrder(order.order_id, confirmToken);
      setOrder(paid);
      const visit = await getVisitGuide(order.order_id);
      setGuide(visit);
      showToast("支付成功，已生成就诊指引");
      track("booking_paid", { order_id: order.order_id, amount: paid.amount });
    } catch (err) {
      showToast(err instanceof Error ? err.message : "支付失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AppShell title="挂号服务" subtitle="选科室、医生与号源，支付后生成就诊指引">
      <div className="mx-auto max-w-4xl space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          {["选号", "锁号", "订单", "支付", "指引"].map((label, index) => (
            <div key={label} className="flex items-center gap-2">
              <span className={clsx("flex h-6 w-6 items-center justify-center rounded-full text-[11px]", index < step ? "bg-emerald-100 text-emerald-600" : index === step ? "bg-primary text-white" : "bg-canvas text-muted")}>
                {index < step ? <CheckCircle2 size={13} /> : index + 1}
              </span>
              <span className={clsx("text-[12px]", index <= step ? "text-ink" : "text-muted")}>{label}</span>
              {index < 4 ? <span className="hidden h-px w-4 bg-line sm:block" /> : null}
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
                        onClick={() => { setSelected(slot); setLock(null); setOrder(null); setGuide(null); }}
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
            <button onClick={doPay} disabled={!order || order.payment_status === "paid" || loading} className="btn-primary"><CreditCard size={15} />确认支付</button>
          </div>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <div className="rounded-2xl bg-canvas p-3 text-[11px] text-muted">
              <p className="font-medium text-ink">当前锁定</p>
              <p className="mt-1 break-all">{lock ? lock.lock_id : "尚未锁定号源"}</p>
            </div>
            <div className="rounded-2xl bg-canvas p-3 text-[11px] text-muted">
              <p className="font-medium text-ink">支付确认（HITL）</p>
              <input className="input mt-1.5 !py-2 text-[12px]" value={confirmToken} onChange={(event) => setConfirmToken(event.target.value)} placeholder="confirm-demo" />
            </div>
          </div>
        </section>

        {order ? (
          <section className="card p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-[11px] text-muted">订单</p>
                <p className="mt-1 text-[15px] font-semibold text-ink">{order.order_id}</p>
              </div>
              <span className={clsx("badge", order.payment_status === "paid" ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700")}>
                {order.payment_status === "paid" ? "已支付" : "待支付"}
              </span>
            </div>
            <p className="mt-2 text-[12px] text-muted">{order.slot_id} · ¥{order.amount}</p>
            {order.payment_status === "paid" && !guide ? (
              <p className="mt-3 flex items-center gap-1.5 text-[12px] text-primary"><Loader2 size={13} className="animate-spin" />正在生成就诊指引…</p>
            ) : null}
            {guide ? (
              <div className="mt-3 rounded-2xl bg-primary-soft p-4 text-[12px] leading-6 text-primary">
                <p className="flex items-center gap-1.5 font-medium"><MapPin size={14} />{guide.hospital} · {guide.floor}</p>
                <p className="mt-1">{guide.address} · {guide.checkin}</p>
                <p className="mt-2 font-medium">携带：</p>
                <div className="mt-1 flex flex-wrap gap-1.5">
                  {guide.items.map((item) => <span key={item} className="rounded-full bg-white/70 px-2 py-0.5 text-[11px]">{item}</span>)}
                </div>
              </div>
            ) : null}
          </section>
        ) : null}

        <div className="flex items-start gap-2 rounded-2xl border border-line bg-card p-3 text-[11px] leading-5 text-muted">
          <ShieldCheck size={14} className="mt-0.5 shrink-0 text-primary" />
          支付前需要人工确认；当前为 mock 医院 MCP，真实环境将替换为医院系统。支付成功后可在健康时间线查看挂号记录与随访待办。
        </div>
      </div>
    </AppShell>
  );
}
