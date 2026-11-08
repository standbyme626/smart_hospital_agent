"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Activity,
  ArrowLeft,
  CalendarClock,
  CheckCircle2,
  CreditCard,
  Loader2,
  Lock,
  ShieldCheck,
  Stethoscope,
} from "lucide-react";
import clsx from "clsx";
import {
  UnauthorizedError,
  createOrder,
  getDepartments,
  getSchedules,
  lockSlot,
  payOrder,
} from "@/lib/api";
import type { BookingOrder, ScheduleSlot } from "@/lib/types";

const steps = ["选择号源", "锁定号源", "创建订单", "支付确认"];

export default function BookingPage() {
  const router = useRouter();
  const [departments, setDepartments] = useState<string[]>([]);
  const [department, setDepartment] = useState("心血管内科");
  const [date, setDate] = useState("");
  const [slots, setSlots] = useState<ScheduleSlot[]>([]);
  const [selected, setSelected] = useState<ScheduleSlot | null>(null);
  const [lock, setLock] = useState<{ lock_id: string; slot_id: string; expires_in: number } | null>(null);
  const [order, setOrder] = useState<BookingOrder | null>(null);
  const [confirmToken, setConfirmToken] = useState("confirm-demo");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleError = useCallback(
    (err: unknown) => {
      if (err instanceof UnauthorizedError) {
        router.push("/login");
        return;
      }
      setError(err instanceof Error ? err.message : "请求失败");
    },
    [router],
  );

  const loadSlots = useCallback(
    async (dept: string, day: string) => {
      setLoading(true);
      setError(null);
      try {
        setSlots(await getSchedules(dept, day || undefined));
      } catch (err) {
        handleError(err);
      } finally {
        setLoading(false);
      }
    },
    [handleError],
  );

  useEffect(() => {
    getDepartments()
      .then((items) => {
        setDepartments(items);
        const first = items.includes("心血管内科") ? "心血管内科" : items[0];
        if (first) {
          setDepartment(first);
          loadSlots(first, "");
        }
      })
      .catch(handleError);
  }, [handleError, loadSlots]);

  const grouped = useMemo(() => {
    const map = new Map<string, ScheduleSlot[]>();
    for (const slot of slots) {
      const list = map.get(slot.date) || [];
      list.push(slot);
      map.set(slot.date, list);
    }
    return Array.from(map.entries()).slice(0, 3);
  }, [slots]);

  const step = order?.payment_status === "paid" ? 4 : order ? 3 : lock ? 2 : selected ? 1 : 0;

  async function doLock() {
    if (!selected) return;
    setLoading(true);
    setError(null);
    try {
      setLock(await lockSlot(selected.slot_id));
    } catch (err) {
      handleError(err);
    } finally {
      setLoading(false);
    }
  }

  async function doOrder() {
    if (!lock) return;
    setLoading(true);
    setError(null);
    try {
      setOrder(await createOrder(lock.lock_id));
    } catch (err) {
      handleError(err);
    } finally {
      setLoading(false);
    }
  }

  async function doPay() {
    if (!order) return;
    setLoading(true);
    setError(null);
    try {
      setOrder(await payOrder(order.order_id, confirmToken));
    } catch (err) {
      handleError(err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-dvh px-4 py-6 sm:px-6">
      <div className="mx-auto max-w-4xl">
        <div className="flex items-center justify-between">
          <Link href="/" className="inline-flex items-center gap-2 text-xs text-slate-400 hover:text-white">
            <ArrowLeft size={14} />
            返回问诊
          </Link>
          <Link href="/" className="flex items-center gap-2 text-sm font-semibold text-white">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-primary2 text-slate-950">
              <Activity size={16} strokeWidth={2.5} />
            </span>
            大医 · 挂号支付
          </Link>
        </div>

        <div className="mt-6 surface p-4 sm:p-6">
          <div className="flex flex-wrap items-center gap-2">
            {steps.map((label, index) => (
              <div key={label} className="flex items-center gap-2">
                <span
                  className={clsx(
                    "flex h-6 w-6 items-center justify-center rounded-full text-[11px]",
                    index < step
                      ? "bg-mint/20 text-mint"
                      : index === step
                        ? "bg-primary text-slate-950"
                        : "bg-panel2 text-slate-500",
                  )}
                >
                  {index < step ? <CheckCircle2 size={13} /> : index + 1}
                </span>
                <span className={clsx("text-xs", index <= step ? "text-slate-200" : "text-slate-500")}>{label}</span>
                {index < steps.length - 1 ? <span className="mx-1 h-px w-6 bg-line" /> : null}
              </div>
            ))}
          </div>

          {error ? <p className="mt-4 rounded-lg border border-rose/30 bg-rose/10 px-3 py-2 text-xs text-rose">{error}</p> : null}

          <div className="mt-6 grid gap-4 sm:grid-cols-2">
            <label className="block text-xs text-slate-400">
              科室
              <select
                className="input mt-1.5"
                value={department}
                onChange={(event) => {
                  const next = event.target.value;
                  setDepartment(next);
                  setSelected(null);
                  setLock(null);
                  setOrder(null);
                  loadSlots(next, date);
                }}
              >
                {departments.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
            <label className="block text-xs text-slate-400">
              日期（可选）
              <input
                className="input mt-1.5"
                type="date"
                value={date}
                onChange={(event) => {
                  const next = event.target.value;
                  setDate(next);
                  setSelected(null);
                  setLock(null);
                  setOrder(null);
                  loadSlots(department, next);
                }}
              />
            </label>
          </div>

          <div className="mt-6">
            <p className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-slate-400">
              <CalendarClock size={14} className="text-primary" />
              可预约号源
            </p>
            {loading ? (
              <div className="mt-4 flex items-center gap-2 text-xs text-slate-400">
                <Loader2 size={14} className="animate-spin" />
                加载中…
              </div>
            ) : (
              <div className="mt-3 space-y-4">
                {grouped.map(([day, items]) => (
                  <div key={day}>
                    <p className="mb-2 text-[11px] text-slate-500">{day}</p>
                    <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                      {items.slice(0, 6).map((slot) => (
                        <button
                          key={slot.slot_id}
                          disabled={!slot.available || Boolean(lock)}
                          onClick={() => {
                            setSelected(slot);
                            setLock(null);
                            setOrder(null);
                          }}
                          className={clsx(
                            "surface-2 px-3 py-2.5 text-left transition",
                            selected?.slot_id === slot.slot_id
                              ? "border-primary/60 ring-2 ring-primary/15"
                              : "hover:border-primary/30",
                            !slot.available ? "opacity-40" : "",
                          )}
                        >
                          <p className="text-sm text-slate-100">
                            {slot.time} · {slot.doctor}
                          </p>
                          <p className="mt-1 text-[11px] text-slate-500">{slot.department} · ¥{slot.fee}</p>
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
                {grouped.length === 0 ? <p className="text-xs text-slate-500">暂无可预约号源。</p> : null}
              </div>
            )}
          </div>

          <div className="mt-6 flex flex-wrap items-center gap-2">
            <button className="btn-ghost" disabled={!selected || Boolean(lock) || loading} onClick={doLock}>
              <Lock size={15} />
              锁定号源
            </button>
            <button className="btn-ghost" disabled={!lock || Boolean(order) || loading} onClick={doOrder}>
              <Stethoscope size={15} />
              创建订单
            </button>
            <button className="btn-primary" disabled={!order || order.payment_status === "paid" || loading} onClick={doPay}>
              <CreditCard size={15} />
              确认支付
            </button>
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <div className="surface-2 px-3 py-2.5 text-xs text-slate-400">
              <p className="flex items-center gap-1.5 text-slate-300">
                <Lock size={12} className="text-primary" />
                当前锁定
              </p>
              <p className="mt-1 break-all">{lock ? lock.lock_id : "尚未锁定号源"}</p>
            </div>
            <div className="surface-2 px-3 py-2.5 text-xs text-slate-400">
              <p className="flex items-center gap-1.5 text-slate-300">
                <ShieldCheck size={12} className="text-mint" />
                支付确认（HITL）
              </p>
              <input
                className="input mt-1.5 !py-1.5 text-xs"
                value={confirmToken}
                onChange={(event) => setConfirmToken(event.target.value)}
                placeholder="confirm-demo"
              />
            </div>
          </div>

          {order ? (
            <div className="mt-4 rounded-xl border border-line bg-panel2 px-3 py-3 text-xs text-slate-300">
              <p className="text-sm text-white">订单 {order.order_id}</p>
              <p className="mt-1 text-slate-400">
                {order.slot_id} · ¥{order.amount} · {order.payment_status === "paid" ? "已支付" : "待支付"}
              </p>
              {order.payment_status === "paid" ? (
                <p className="mt-2 inline-flex items-center gap-1.5 rounded-lg border border-mint/40 bg-mint/10 px-2 py-1 text-mint">
                  <CheckCircle2 size={13} />
                  挂号成功（mock MCP），真实环境将调用医院系统
                </p>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
