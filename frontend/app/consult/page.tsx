"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Check,
  CheckCircle2,
  Loader2,
  Send,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  UserRound,
} from "lucide-react";
import clsx from "clsx";
import { AppShell } from "@/components/AppShell";
import { ResultView } from "@/components/ResultView";
import { useApp } from "@/lib/app-context";
import { getSession, streamChat, track } from "@/lib/api";
import { memberMismatch } from "@/lib/session-guard";
import type { ChatMessage, Citation, Report, Safety, StreamEvent, Triage } from "@/lib/types";
import {
  ageFromBirthDate,
  bodyAreas,
  durationOptions,
  redFlagOptions,
  stripMemberContext,
  symptomOptions,
} from "@/lib/format";

type Step = 0 | 1 | 2;
type SymptomStep = 0 | 1 | 2 | 3 | 4;

function displayContent(content: string): string {
  return stripMemberContext(content);
}

export default function ConsultPage() {
  const router = useRouter();
  const { authReady, members, activeMember, setActiveMemberId, showToast } = useApp();
  const [step, setStep] = useState<Step>(0);
  const [memberId, setMemberId] = useState<string | null>(activeMember?.id || null);
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");
  const [draft, setDraft] = useState("");
  const [symptomStep, setSymptomStep] = useState<SymptomStep>(0);
  const [bodyArea, setBodyArea] = useState("");
  const [symptoms, setSymptoms] = useState<string[]>([]);
  const [duration, setDuration] = useState("");
  const [severity, setSeverity] = useState(3);
  const [redFlag, setRedFlag] = useState("");
  const [allergies, setAllergies] = useState("");
  const [conditions, setConditions] = useState("");
  const [medications, setMedications] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [statusLines, setStatusLines] = useState<string[]>([]);
  const [streamingText, setStreamingText] = useState("");
  const [report, setReport] = useState<Report | null>(null);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [safety, setSafety] = useState<Safety | undefined>();
  const [triage, setTriage] = useState<Triage | undefined>();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionMemberId, setSessionMemberId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [followup, setFollowup] = useState("");
  const [restoring, setRestoring] = useState(false);

  useEffect(() => {
    const saved = window.sessionStorage.getItem("consult_draft");
    if (saved) {
      setDraft(saved);
      window.sessionStorage.removeItem("consult_draft");
    }
  }, []);

  useEffect(() => {
    if (!authReady) return;
    const existing = new URLSearchParams(window.location.search).get("session");
    if (!existing) return;
    let cancelled = false;
    setRestoring(true);
    (async () => {
      try {
        const detail = await getSession(existing);
        if (cancelled) return;
        if (memberMismatch(detail.session.member_id, activeMember?.id)) {
          reset();
          showToast("当前会话属于其他成员，已停止恢复，请重新开始");
          return;
        }
        setSessionId(detail.session.id);
        setSessionMemberId(detail.session.member_id || null);
        if (detail.session.member_id) {
          setMemberId(detail.session.member_id);
          setActiveMemberId(detail.session.member_id);
        }
        setMessages(detail.messages || []);
        const latest = detail.latest_final;
        if (latest?.report) {
          setReport(latest.report);
          setCitations(latest.citations || []);
          setSafety(latest.safety);
          setTriage(latest.triage);
          setStep(2);
        }
      } catch {
        if (!cancelled) showToast("会话恢复失败，请重新开始");
      } finally {
        if (!cancelled) setRestoring(false);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authReady]);

  useEffect(() => {
    if (activeMember) {
      setMemberId(activeMember.id);
      setAge(ageFromBirthDate(activeMember.birth_date)?.toString() ?? "");
      setGender(activeMember.gender || "");
      setAllergies(activeMember.allergies || "");
      setConditions(activeMember.conditions || "");
    }
  }, [activeMember]);

  const member = useMemo(() => members.find((item) => item.id === memberId) || activeMember, [members, memberId, activeMember]);
  const memberAge = member?.birth_date ? ageFromBirthDate(member.birth_date) : null;

  function toggleSymptom(value: string) {
    setSymptoms((prev) => (prev.includes(value) ? prev.filter((item) => item !== value) : [...prev, value]));
  }

  function applyStreamEvent(event: StreamEvent) {
    if (event.type === "session") {
      const id = String(event.data.session_id || "");
      if (id) {
        setSessionId(id);
        window.history.replaceState(null, "", `/consult/?session=${encodeURIComponent(id)}`);
      }
    } else if (event.type === "status") {
      const stage = String(event.data.stage || "");
      const detail = event.data.detail as Record<string, unknown> | undefined;
      const label =
        stage === "triage" ? "正在分诊" :
        stage === "triage_done" ? `分诊完成 · ${detail?.department || ""}` :
        stage === "tool" ? `检查 · ${event.data.tool || ""}` :
        stage === "drug_check" ? `用药检查 · ${event.data.highest_severity || "none"}` : stage;
      setStatusLines((prev) => [...prev, label]);
    } else if (event.type === "thought") {
      const text = String(event.data.text || "").trim();
      if (text) setStatusLines((prev) => [...prev, text]);
    } else if (event.type === "citation") {
      const citation = event.data as unknown as Citation;
      setCitations((prev) => (prev.some((item) => String(item.id) === String(citation.id)) ? prev : [...prev, citation]));
    } else if (event.type === "token") {
      setStreamingText((prev) => prev + String(event.data.text || ""));
    } else if (event.type === "final") {
      const nextReport = event.data.report as Report;
      setReport(nextReport);
      setCitations((event.data.citations as unknown as Citation[]) || []);
      setSafety(event.data.safety as Safety);
      setTriage(event.data.triage as Triage);
      setStep(2);
      if (nextReport?.summary) {
        setMessages((prev) => [...prev, { role: "assistant", content: nextReport.summary }]);
      }
      track("consult_complete", { department: nextReport?.department });
    } else if (event.type === "error") {
      showToast(String(event.data.message || "服务暂时不可用"));
    }
  }

  async function submit() {
    if (!bodyArea && !draft.trim()) {
      showToast("请先描述哪里不舒服");
      setStep(1);
      setSymptomStep(0);
      return;
    }
    setSubmitting(true);
    setStatusLines([]);
    setStreamingText("");
    setReport(null);
    setCitations([]);
    track("consult_submit", { member_id: memberId, body_area: bodyArea, symptoms: symptoms.join(",") });
    const symptomBits = [
      !member?.birth_date && age ? `年龄：${age}` : "",
      bodyArea ? `主要部位：${bodyArea}` : "",
      symptoms.length ? `症状：${symptoms.join("、")}` : "",
      duration ? `持续时间：${duration}` : "",
      `严重程度：${severity}/5`,
      redFlag ? `红旗筛查：${redFlagOptions.find((item) => item.value === redFlag)?.label}` : "",
      allergies ? `过敏史：${allergies}` : "",
      conditions ? `既往史：${conditions}` : "",
      medications ? `正在用药：${medications}` : "",
      draft ? `补充描述：${draft}` : "",
    ].filter(Boolean);
    const message = symptomBits.join("\n");
    const title = [bodyArea, symptoms.join("、"), duration].filter(Boolean).join(" · ") || draft.trim().slice(0, 40) || "问诊";
    setSessionMemberId(memberId || null);
    setMessages((prev) => [...prev, { role: "user", content: message }]);
    try {
      for await (const event of streamChat({ message, title, session_id: sessionId || undefined, member_id: memberId || undefined })) {
        applyStreamEvent(event);
      }
    } catch (err) {
      showToast(err instanceof Error ? err.message : "请求失败，请稍后再试");
    } finally {
      setSubmitting(false);
    }
  }

  async function sendFollowup() {
    const text = followup.trim();
    if (!text || submitting || !sessionId) return;
    if (memberMismatch(sessionMemberId, activeMember?.id)) {
      showToast("当前会话属于其他成员，已停止追问；请重新开始会话");
      return;
    }
    setFollowup("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setSubmitting(true);
    setStatusLines([]);
    setStreamingText("");
    track("consult_followup", { session_id: sessionId });
    try {
      for await (const event of streamChat({ message: text, session_id: sessionId, member_id: memberId || undefined })) {
        applyStreamEvent(event);
      }
    } catch (err) {
      showToast(err instanceof Error ? err.message : "请求失败，请稍后再试");
    } finally {
      setSubmitting(false);
    }
  }

  function reset() {
    setStep(0);
    setSymptomStep(0);
    setBodyArea("");
    setSymptoms([]);
    setDuration("");
    setSeverity(3);
    setRedFlag("");
    setDraft("");
    setReport(null);
    setCitations([]);
    setStatusLines([]);
    setStreamingText("");
    setMessages([]);
    setFollowup("");
    setSessionId(null);
    setSessionMemberId(null);
    setAge(ageFromBirthDate(activeMember?.birth_date)?.toString() ?? "");
    setGender(activeMember?.gender || "");
    setAllergies(activeMember?.allergies || "");
    setConditions(activeMember?.conditions || "");
    window.history.replaceState(null, "", "/consult/");
  }

  function nextSymptom() {
    if (symptomStep < 4) setSymptomStep((value) => (value + 1) as SymptomStep);
    else setStep(2);
  }

  return (
    <AppShell title="AI 诊室" subtitle="一屏一问，先分诊再决定下一步">
      <div className="mx-auto max-w-3xl space-y-4">
        <div className="flex items-center gap-2">
          {["就诊人", "症状", "结果"].map((label, index) => (
            <div key={label} className="flex flex-1 items-center gap-2">
              <span className={clsx("flex h-6 w-6 items-center justify-center rounded-full text-[11px]", index <= step ? "bg-primary text-white" : "bg-canvas text-muted")}>
                {index < step ? <Check size={13} /> : index + 1}
              </span>
              <span className={clsx("text-[12px]", index <= step ? "text-ink" : "text-muted")}>{label}</span>
              {index < 2 ? <span className="h-px flex-1 bg-line" /> : null}
            </div>
          ))}
        </div>

        {step === 0 ? (
          <div className="card p-5 sm:p-6 fade-up">
            <div className="flex items-center gap-2 text-primary"><Stethoscope size={18} /><h2 className="text-[17px] font-semibold">这次为谁咨询？</h2></div>
            <p className="mt-1 text-[12px] text-muted">选择家庭成员后，我们会带上对应档案信息。</p>
            <div className="mt-4 flex flex-wrap gap-2">
              {members.map((item) => (
                <button
                  key={item.id}
                  onClick={() => {
                    setMemberId(item.id);
                    setActiveMemberId(item.id);
                    setAge(ageFromBirthDate(item.birth_date)?.toString() ?? "");
                    setGender(item.gender || "");
                    setAllergies(item.allergies || "");
                    setConditions(item.conditions || "");
                  }}
                  className={clsx("flex min-h-11 items-center gap-2 rounded-full border px-3 text-[13px]", memberId === item.id ? "border-primary bg-primary-soft text-primary" : "border-line bg-white text-ink")}
                >
                  <UserRound size={14} />
                  {item.name}
                  <span className="text-[11px] text-muted">{item.relation === "self" ? "本人" : item.relation}</span>
                </button>
              ))}
              <button onClick={() => router.push("/profile/")} className="chip text-primary">+ 添加家人</button>
            </div>

            <div className="mt-6 grid gap-4 sm:grid-cols-2">
              <label className="block">
                <span className="label">性别</span>
                <select className="input" value={gender} onChange={(event) => setGender(event.target.value)}>
                  <option value="">未填写</option>
                  <option value="男">男</option>
                  <option value="女">女</option>
                  <option value="其他">其他</option>
                </select>
              </label>
              <label className="block">
                <span className="label">年龄</span>
                {memberAge !== null ? (
                  <>
                    <input className="input" value={String(memberAge)} readOnly aria-readonly="true" />
                    <p className="mt-1 text-[11px] text-muted">
                      来自家庭档案出生日期 · <a className="text-primary underline underline-offset-2" href="/profile/">去修改</a>
                    </p>
                  </>
                ) : (
                  <>
                    <input className="input" value={age} onChange={(event) => setAge(event.target.value)} placeholder="例如 56" inputMode="numeric" />
                    <p className="mt-1 text-[11px] text-muted">在设置与家庭填写出生日期后自动带出</p>
                  </>
                )}
              </label>
            </div>

            <div className="mt-6 flex justify-end">
              <button className="btn-primary" onClick={() => setStep(1)}>
                下一步
                <ArrowRight size={15} />
              </button>
            </div>
          </div>
        ) : null}

        {step === 1 ? (
          <div className="card p-5 sm:p-6 fade-up">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-primary"><Sparkles size={18} /><h2 className="text-[17px] font-semibold">哪里不舒服？</h2></div>
              <span className="text-[11px] text-muted">{symptomStep + 1} / 5</span>
            </div>

            {symptomStep === 0 ? (
              <div className="mt-5">
                <p className="text-[15px] font-medium text-ink">主要是哪个部位？</p>
                <div className="mt-3 grid grid-cols-3 gap-2 sm:grid-cols-4">
                  {bodyAreas.map((item) => (
                    <button key={item} onClick={() => { setBodyArea(item); nextSymptom(); }} className={clsx("min-h-12 rounded-2xl border px-2 text-[13px]", bodyArea === item ? "border-primary bg-primary-soft text-primary" : "border-line bg-white text-ink hover:border-primary/30")}>
                      {item}
                    </button>
                  ))}
                </div>
                <button onClick={nextSymptom} className="mt-4 text-[12px] text-muted underline underline-offset-2">不确定，跳过</button>
              </div>
            ) : null}

            {symptomStep === 1 ? (
              <div className="mt-5">
                <p className="text-[15px] font-medium text-ink">有哪些症状？可以多选</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {symptomOptions.map((item) => (
                    <button key={item} onClick={() => toggleSymptom(item)} className={clsx("chip min-h-11 px-4", symptoms.includes(item) ? "chip-active" : "")}>
                      {symptoms.includes(item) ? <Check size={13} /> : null}
                      {item}
                    </button>
                  ))}
                </div>
                <label className="mt-4 block">
                  <span className="label">再补充一句（可选）</span>
                  <textarea className="input min-h-[84px]" value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="例如：两天了，晚上更明显，吃退烧药会退但反复…" />
                </label>
              </div>
            ) : null}

            {symptomStep === 2 ? (
              <div className="mt-5">
                <p className="text-[15px] font-medium text-ink">持续多久了？</p>
                <div className="mt-3 grid grid-cols-2 gap-2">
                  {durationOptions.map((item) => (
                    <button key={item} onClick={() => { setDuration(item); nextSymptom(); }} className={clsx("min-h-12 rounded-2xl border px-3 text-[13px]", duration === item ? "border-primary bg-primary-soft text-primary" : "border-line bg-white text-ink hover:border-primary/30")}>
                      {item}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}

            {symptomStep === 3 ? (
              <div className="mt-5">
                <p className="text-[15px] font-medium text-ink">现在有多难受？</p>
                <div className="mt-4 grid grid-cols-5 gap-2">
                  {[1, 2, 3, 4, 5].map((value) => (
                    <button key={value} onClick={() => setSeverity(value)} className={clsx("min-h-12 rounded-2xl border text-[15px] font-semibold", severity === value ? "border-primary bg-primary-soft text-primary" : "border-line bg-white text-ink")}>
                      {value}
                    </button>
                  ))}
                </div>
                <div className="mt-2 flex justify-between text-[11px] text-muted"><span>轻微</span><span>非常严重</span></div>
              </div>
            ) : null}

            {symptomStep === 4 ? (
              <div className="mt-5">
                <p className="flex items-center gap-2 text-[15px] font-medium text-ink"><AlertTriangle size={17} className="text-amber-500" />有没有以下危险信号？</p>
                <p className="mt-1 text-[12px] leading-5 text-muted">胸痛伴出汗、呼吸困难、意识不清、大出血、抽搐、剧烈头痛伴颈项强直等。</p>
                <div className="mt-3 space-y-2">
                  {redFlagOptions.map((item) => (
                    <button key={item.value} onClick={() => setRedFlag(item.value)} className={clsx("flex min-h-12 w-full items-center gap-2 rounded-2xl border px-3 text-left text-[13px]", redFlag === item.value ? "border-primary bg-primary-soft text-primary" : "border-line bg-white text-ink")}>
                      <span className={clsx("flex h-4 w-4 items-center justify-center rounded-full border", redFlag === item.value ? "border-primary bg-primary" : "border-line")}>{redFlag === item.value ? <Check size={10} className="text-white" /> : null}</span>
                      {item.label}
                    </button>
                  ))}
                </div>
                <div className="mt-5 flex gap-2">
                  <button onClick={() => setStep(0)} className="btn-ghost flex-1"><ArrowLeft size={15} />上一步</button>
                  <button onClick={() => setStep(2)} className="btn-primary flex-1">补充病史<ArrowRight size={15} /></button>
                </div>
              </div>
            ) : null}

            {symptomStep > 0 && symptomStep < 4 ? (
              <div className="mt-5 flex gap-2">
                <button onClick={() => setSymptomStep((value) => (value - 1) as SymptomStep)} className="btn-ghost flex-1"><ArrowLeft size={15} />上一步</button>
                <button onClick={nextSymptom} className="btn-primary flex-1">下一步<ArrowRight size={15} /></button>
              </div>
            ) : null}
          </div>
        ) : null}

        {step === 2 && !report ? (
          <div className="card p-5 sm:p-6 fade-up">
            <div className="flex items-center gap-2 text-primary"><ShieldCheck size={18} /><h2 className="text-[17px] font-semibold">补充关键信息</h2></div>
            <p className="mt-1 text-[12px] text-muted">越完整，分诊越准确；没有也可以直接开始。</p>
            <div className="mt-4 space-y-3">
              <label className="block"><span className="label">过敏史</span><textarea className="input min-h-[72px]" value={allergies} onChange={(event) => setAllergies(event.target.value)} placeholder="例如：青霉素过敏" /></label>
              <label className="block"><span className="label">既往史 / 慢病</span><textarea className="input min-h-[72px]" value={conditions} onChange={(event) => setConditions(event.target.value)} placeholder="例如：高血压 10 年，糖尿病" /></label>
              <label className="block"><span className="label">正在用药</span><textarea className="input min-h-[72px]" value={medications} onChange={(event) => setMedications(event.target.value)} placeholder="例如：氨氯地平 5mg 每日一次" /></label>
            </div>
            <div className="mt-5 flex gap-2">
              <button onClick={() => { setStep(1); setSymptomStep(4); }} className="btn-ghost flex-1"><ArrowLeft size={15} />上一步</button>
              <button onClick={submit} disabled={submitting} className="btn-primary flex-1">
                {submitting ? <Loader2 size={15} className="animate-spin" /> : <Stethoscope size={15} />}
                {submitting ? "分诊中…" : "开始分诊"}
              </button>
            </div>
          </div>
        ) : null}

        {submitting ? (
          <div className="card p-4">
            <div className="flex items-center gap-2 text-[13px] font-medium text-ink"><Loader2 size={15} className="animate-spin text-primary" />大医正在整理</div>
            <div className="mt-2 space-y-1">
              {statusLines.slice(-4).map((line, index) => (
                <p key={index} className="flex items-center gap-1.5 text-[12px] text-muted"><CheckCircle2 size={12} className="text-primary" />{line}</p>
              ))}
            </div>
            {streamingText ? <p className="mt-3 whitespace-pre-wrap text-[13px] leading-6 text-ink">{streamingText}</p> : null}
          </div>
        ) : null}

        {restoring ? (
          <div className="card flex items-center gap-2 p-4 text-[13px] text-muted">
            <Loader2 size={15} className="animate-spin text-primary" />
            正在恢复上次会话…
          </div>
        ) : null}

        {report ? (
          <div className="fade-up space-y-4">
            <ResultView
              report={report}
              citations={citations}
              safety={safety}
              triage={triage}
              onBook={() => {
                window.sessionStorage.setItem("booking_department", report.department);
                router.push("/services/");
              }}
              onRestart={reset}
            />
            {messages.length ? (
              <div className="card p-4">
                <h3 className="section-title">对话记录</h3>
                <div className="mt-3 space-y-2">
                  {messages.map((item, index) => (
                    <div
                      key={index}
                      className={clsx(
                        "rounded-2xl px-3 py-2 text-[13px] leading-6",
                        item.role === "user" ? "ml-8 bg-primary-soft text-ink" : "mr-8 bg-canvas text-ink",
                      )}
                    >
                      <p className="text-[11px] text-muted">{item.role === "user" ? "我" : "大医"}</p>
                      <p className="whitespace-pre-wrap">
                        {item.role === "user" ? displayContent(item.content) : item.content}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
            <div className="card p-4">
              <h3 className="section-title">继续追问</h3>
              <p className="mt-1 text-[12px] text-muted">追问会沿用本次会话，大医能看到前面的问答。</p>
              <div className="mt-3 flex items-end gap-2">
                <textarea
                  className="input min-h-[72px] flex-1"
                  value={followup}
                  onChange={(event) => setFollowup(event.target.value)}
                  placeholder={report.followup_question ? `回答：${report.followup_question}` : "继续问，例如：需要做哪些检查？"}
                />
                <button
                  onClick={sendFollowup}
                  disabled={submitting || !followup.trim()}
                  className="btn-primary h-11 px-3"
                  aria-label="发送追问"
                >
                  {submitting ? <Loader2 size={15} className="animate-spin" /> : <Send size={16} />}
                </button>
              </div>
            </div>
            {sessionId ? <p className="text-center text-[11px] text-muted">会话已保存 · {sessionId}</p> : null}
          </div>
        ) : null}
      </div>
    </AppShell>
  );
}
