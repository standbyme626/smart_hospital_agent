"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowRight,
  BellRing,
  CalendarClock,
  Camera,
  CheckCircle2,
  ClipboardList,
  ChevronRight,
  Clock,
  FileText,
  HeartPulse,
  Mic,
  Pill,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  Users,
} from "lucide-react";
import Link from "next/link";
import { AppShell } from "@/components/AppShell";
import { TaskCard } from "@/components/TaskCard";
import { useApp } from "@/lib/app-context";
import { completeFollowup, getHomeSummary, track } from "@/lib/api";
import { formatDate, formatSessionTitle, kindMeta, relativeTime, timelineHref } from "@/lib/format";
import type { HomeSummary } from "@/lib/types";

export default function HomePage() {
  const router = useRouter();
  const { activeMember, showToast, refreshFamily } = useApp();
  const [summary, setSummary] = useState<HomeSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);
  const [draft, setDraft] = useState("");

  async function load() {
    setLoading(true);
    setLoadError(false);
    setSummary(null);
    try {
      const data = await getHomeSummary(activeMember?.id);
      setSummary(data);
    } catch {
      setSummary(null);
      setLoadError(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    setLoading(true);
    load();
    track("home_view", { member_id: activeMember?.id || null });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeMember?.id]);

  function startConsult(text?: string) {
    const value = (text ?? draft).trim();
    if (value) window.sessionStorage.setItem("consult_draft", value);
    track("consult_start", { from: "home", has_text: Boolean(value) });
    router.push("/consult/");
  }

  function startVoice() {
    const SpeechRecognition = (window as unknown as { SpeechRecognition?: any; webkitSpeechRecognition?: any }).SpeechRecognition
      || (window as unknown as { webkitSpeechRecognition?: any }).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      showToast("当前浏览器不支持语音输入，请先用文字描述");
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = "zh-CN";
    recognition.interimResults = false;
    recognition.onresult = (event: any) => {
      const text = event.results?.[0]?.[0]?.transcript || "";
      if (text) setDraft((prev) => (prev ? `${prev} ${text}` : text));
    };
    recognition.onerror = () => showToast("没有听清，请再试一次");
    recognition.start();
  }

  async function finishFollowup(id: string) {
    try {
      await completeFollowup(id);
      showToast("已完成随访");
      load();
    } catch {
      showToast("操作失败，请稍后再试");
    }
  }

  const hour = new Date().getHours();
  const greeting = hour < 11 ? "早上好" : hour < 18 ? "下午好" : "晚上好";
  const memberName = activeMember?.name || "本人";

  return (
    <AppShell
      title={`${greeting}，${memberName}`}
      subtitle="今天想先解决哪件事？"
      actions={
        <div className="flex items-center gap-2">
          <span className="hidden rounded-full bg-primary-soft px-3 py-1 text-[11px] text-primary sm:inline-flex">AI 分诊 · 证据可查</span>
          <Link href="/messages/" className="relative flex h-9 w-9 items-center justify-center rounded-full bg-white text-muted ring-1 ring-line hover:text-primary" aria-label="消息">
            <BellRing size={16} />
            {summary?.followups?.length ? <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-rose-500" /> : null}
          </Link>
        </div>
      }
    >
      <div className="space-y-5">
        {loadError ? (
          <div className="card flex flex-wrap items-center justify-between gap-2 border-rose-200 bg-rose-50 p-3 text-[13px] text-rose-700">
            <span>加载失败，请重试</span>
            <button onClick={load} className="min-h-11 rounded-full bg-white px-4 text-[12px] font-medium text-rose-700">重试</button>
          </div>
        ) : null}
        <section className="card overflow-hidden p-5 sm:p-7">
          <div className="flex items-center gap-2 text-[12px] text-muted">
            <Sparkles size={14} className="text-primary" />
            为 {memberName} 咨询
            <span className="text-line">·</span>
            <span>不确定挂什么科？先说说哪里不舒服</span>
          </div>
          <h1 className="mt-3 text-[26px] font-semibold leading-tight text-ink sm:text-[32px]">
            哪里不舒服？
          </h1>
          <form
            className="mt-4"
            onSubmit={(event) => {
              event.preventDefault();
              startConsult();
            }}
          >
            <div className="rounded-2xl border border-line bg-canvas p-2 focus-within:border-primary/50 focus-within:ring-4 focus-within:ring-primary/10">
              <textarea
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                rows={2}
                placeholder="例如：头痛伴发热两天，吃了退烧药还是反复…"
                className="min-h-[54px] w-full resize-none bg-transparent px-2 py-1.5 text-[15px] leading-6 text-ink outline-none placeholder:text-muted/70"
              />
              <div className="flex items-center justify-between px-1 pb-1">
                <div className="flex items-center gap-1">
                  <button type="button" onClick={() => router.push("/reports/")} className="flex h-9 w-9 items-center justify-center rounded-full text-muted hover:bg-white hover:text-primary" aria-label="上传报告">
                    <Camera size={17} />
                  </button>
                  <button type="button" onClick={startVoice} className="flex h-9 w-9 items-center justify-center rounded-full text-muted hover:bg-white hover:text-primary" aria-label="浏览器语音输入">
                    <Mic size={17} />
                  </button>
                </div>
                <button type="submit" className="btn-primary min-h-10 px-4">
                  开始问诊
                  <ArrowRight size={15} />
                </button>
              </div>
            </div>
          </form>
          <div className="mt-3 flex flex-wrap gap-2">
            {["头痛发热", "血压偏高", "孩子发烧", "体检报告解读"].map((item) => (
              <button
                key={item}
                onClick={() => {
                  if (item === "体检报告解读") router.push("/reports/");
                  else startConsult(item);
                }}
                className="chip hover:border-primary/40 hover:text-primary"
              >
                {item}
              </button>
            ))}
          </div>
        </section>

        <section>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="section-title">我想…</h2>
            <span className="text-[11px] text-muted">选一个入口，少走弯路</span>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <TaskCard href="/consult/" icon={Stethoscope} title="AI 诊室" description="描述症状，得到分诊与就医方向" />
            <TaskCard href="/reports/" icon={FileText} title="报告解读" description="上传化验/检查报告，看懂异常指标" tone="amber" />
            <TaskCard href="/meds/" icon={Pill} title="药管家" description="用药清单与相互作用检查" tone="emerald" />
            <TaskCard href="/previsit/" icon={ClipboardList} title="就诊准备" description="提前写清主诉与用药，医生接诊直接看" tone="amber" />
            <TaskCard href="/services/" icon={CalendarClock} title="挂号服务" description="选科室、医生与号源，支付由管理员审核" tone="sky" />
            <TaskCard href="/health/" icon={HeartPulse} title="健康档案" description="报告、用药、就诊与随访时间线" tone="violet" />
          </div>
        </section>

        <div className="flex flex-wrap gap-2">
          {[
            { href: "/pickup/", label: "取药进度" },
            { href: "/insurance/", label: "医保结算" },
            { href: "/queue/", label: "排队叫号" },
            { href: "/support/", label: "客服反馈" },
            { href: "/consent/", label: "家庭授权" },
          ].map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="rounded-full bg-white px-3 py-1.5 text-[12px] text-muted ring-1 ring-line hover:text-primary"
            >
              {item.label}
            </Link>
          ))}
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <section className="card p-5">
            <div className="flex items-center justify-between">
              <h2 className="section-title">继续上次</h2>
              {summary?.latest_session ? <span className="text-[11px] text-muted">{relativeTime(summary.latest_session.updated_at)}</span> : null}
            </div>
            {loading ? (
              <div className="mt-4 h-16 animate-pulse rounded-2xl bg-canvas" />
            ) : summary?.latest_session ? (
              <button onClick={() => router.push(`/consult/?session=${encodeURIComponent(summary.latest_session!.id)}`)} className="mt-4 flex w-full items-center justify-between gap-3 rounded-2xl bg-canvas p-3 text-left">
                <span className="min-w-0">
                  <span className="block truncate text-[14px] font-medium text-ink">{formatSessionTitle(summary.latest_session.title)}</span>
                  <span className="mt-1 block text-[11px] text-muted">回到上次问诊，继续追问</span>
                </span>
                <ChevronRight size={16} className="shrink-0 text-muted" />
              </button>
            ) : !loadError ? (
              <div className="mt-4 rounded-2xl bg-canvas p-4 text-[13px] text-muted">
                还没有问诊记录。说说哪里不舒服，大医帮你先分诊。
              </div>
            ) : null}
          </section>

          <section className="card p-5">
            <div className="flex items-center justify-between">
              <h2 className="section-title">今日待办</h2>
              <span className="text-[11px] text-muted">{loading || loadError || !summary ? "—" : `${summary.followups.length + summary.medications.length} 项`}</span>
            </div>
            <div className="mt-3 space-y-2">
              {loading ? <div className="h-16 animate-pulse rounded-2xl bg-canvas" /> : null}
              {summary?.followups?.slice(0, 2).map((item) => (
                <div key={item.id} className="flex items-center justify-between gap-2 rounded-2xl bg-canvas p-3">
                  <span className="min-w-0">
                    <span className="block truncate text-[13px] font-medium text-ink">{item.title}</span>
                    <span className="mt-0.5 flex items-center gap-1 text-[11px] text-muted"><Clock size={11} /> {item.due_date ? formatDate(item.due_date) : "待安排"}</span>
                  </span>
                  <button onClick={() => finishFollowup(item.id)} className="flex h-11 w-11 items-center justify-center rounded-full bg-white text-primary hover:bg-primary-soft" aria-label="完成随访">
                    <CheckCircle2 size={15} />
                  </button>
                </div>
              ))}
              {summary?.medications?.slice(0, 2).map((item) => (
                <div key={item.id} className="flex items-center gap-3 rounded-2xl bg-canvas p-3">
                  <span className="flex h-8 w-8 items-center justify-center rounded-full bg-emerald-50 text-emerald-600"><Pill size={15} /></span>
                  <span className="min-w-0">
                    <span className="block truncate text-[13px] font-medium text-ink">{item.name}</span>
                    <span className="mt-0.5 block text-[11px] text-muted">{item.dose || ""} {item.frequency || ""}</span>
                  </span>
                </div>
              ))}
              {!loading && !loadError && !summary?.followups?.length && !summary?.medications?.length ? (
                <div className="rounded-2xl bg-canvas p-4 text-[13px] text-muted">今天没有待办，保持好作息。</div>
              ) : null}
            </div>
          </section>
        </div>

        <section className="card p-5">
          <div className="flex items-center justify-between">
            <h2 className="section-title">最近记录</h2>
            <button onClick={() => router.push("/health/")} className="text-[12px] text-primary">查看全部</button>
          </div>
          <div className="mt-3 space-y-2">
            {loading ? <div className="h-16 animate-pulse rounded-2xl bg-canvas" /> : null}
            {summary?.timeline?.slice(0, 3).map((event) => {
              const meta = kindMeta[event.kind] || kindMeta.consult;
              const href = timelineHref(event.kind, event.detail);
              const content = (
                <>
                  <span className={`badge ${meta.color}`}>{meta.label}</span>
                  <span className="min-w-0 flex-1 truncate text-[13px] text-ink">{event.title}</span>
                  <span className="shrink-0 text-[11px] text-muted">{relativeTime(event.occurred_at)}</span>
                </>
              );
              return href ? (
                <Link key={event.id} href={href} className="flex items-center gap-3 rounded-2xl bg-canvas p-3 transition hover:bg-primary-soft/50">
                  {content}
                  <ChevronRight size={14} className="shrink-0 text-muted" />
                </Link>
              ) : (
                <div key={event.id} className="flex items-center gap-3 rounded-2xl bg-canvas p-3">{content}</div>
              );
            })}
            {!loading && !loadError && !summary?.timeline?.length ? (
              <p className="rounded-2xl bg-canvas p-4 text-[13px] text-muted">你的问诊、报告、用药和挂号记录会显示在这里。</p>
            ) : null}
          </div>
        </section>

        <section className="grid gap-3 sm:grid-cols-3">
          {[
            { icon: ShieldCheck, title: "证据可追溯", text: "关键建议附本地知识库依据（如有）" },
            { icon: Users, title: "家庭档案", text: "为老人孩子分别建档" },
            { icon: CalendarClock, title: "服务闭环", text: "分诊 → 挂号 → 健康档案" },
          ].map((item) => (
            <div key={item.title} className="flex items-center gap-3 rounded-2xl border border-line bg-card p-3">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary-soft text-primary"><item.icon size={16} /></span>
              <span>
                <span className="block text-[12px] font-medium text-ink">{item.title}</span>
                <span className="block text-[11px] text-muted">{item.text}</span>
              </span>
            </div>
          ))}
        </section>
      </div>
    </AppShell>
  );
}
