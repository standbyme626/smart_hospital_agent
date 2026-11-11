"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Cpu, Database, SlidersHorizontal } from "lucide-react";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { PlatformNav } from "@/components/PlatformNav";
import {
  getCompliance,
  getEvaluation,
  getHealthSummary,
  getPlatformOverview,
  type ComplianceInfo,
  type EvaluationData,
  type HealthSummary,
  type PlatformOverview,
} from "@/lib/api";

type Tab = "overview" | "config" | "capabilities" | "compliance" | "health" | "evaluation";

const COUNT_LABEL: Record<string, string> = {
  users: "用户",
  sessions: "会话",
  reports: "报告",
  followups: "随访",
  notifications: "通知",
  documents: "知识文档",
  pending_reviews: "待审核",
  open_findings: "对账差异",
};

export default function PlatformPage() {
  const [tab, setTab] = useState<Tab>("overview");
  const [data, setData] = useState<PlatformOverview | null>(null);
  const [compliance, setCompliance] = useState<ComplianceInfo | null>(null);
  const [health, setHealth] = useState<HealthSummary | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [denied, setDenied] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setFailed(false);
    setDenied(false);
    try {
      const [overview, complianceData, healthData, evaluationData] = await Promise.all([
        getPlatformOverview(),
        getCompliance().catch(() => null),
        getHealthSummary().catch(() => null),
        getEvaluation().catch(() => null),
      ]);
      setData(overview);
      setCompliance(complianceData);
      setHealth(healthData);
      setEvaluation(evaluationData);
    } catch (error) {
      const status = (error as { status?: number } | null)?.status;
      if (status === 401 || status === 403) setDenied(true);
      else setFailed(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="min-h-dvh bg-canvas">
      <header className="sticky top-0 z-20 border-b border-line bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-5xl items-center gap-3 px-4 py-3">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary-soft text-primary">
            <SlidersHorizontal size={17} />
          </span>
          <div className="min-w-0">
            <p className="text-[15px] font-semibold text-ink">平台管理</p>
            <p className="text-[11px] text-muted">运行总览 / 配置管理 / 知识库 / 可观测 / 接入与实验 / 规范与数据流</p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <SourceBadge live />
            <Link href="/" className="inline-flex min-h-9 items-center gap-1 rounded-full bg-white px-3 text-[12px] text-muted ring-1 ring-line hover:text-primary">
              <ArrowLeft size={13} /> 患者端
            </Link>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl space-y-4 px-4 py-5">
        <PlatformNav />
        <OfflineBanner />

        {denied ? (
          <div className="card p-3">
            <StateBlock kind="denied" title="当前账号无管理权限" description="平台管理只对 admin 角色开放" />
            <div className="mt-2 text-center"><Link href="/login/" className="text-[12px] font-medium text-primary">去登录</Link></div>
          </div>
        ) : (
          <>
            <div className="flex flex-wrap gap-2">
              {([
                { key: "overview", label: "运行总览" },
                { key: "config", label: "开关与配置" },
                { key: "capabilities", label: "能力矩阵" },
                { key: "compliance", label: "合规与数据" },
                { key: "health", label: "告警与 SLO" },
                { key: "evaluation", label: "质量评测" },
              ] as Array<{ key: Tab; label: string }>).map((item) => (
                <button
                  key={item.key}
                  onClick={() => setTab(item.key)}
                  className={tab === item.key ? "chip border-primary bg-primary text-white" : "chip hover:border-primary/40 hover:text-primary"}
                >
                  {item.label}
                </button>
              ))}
            </div>

            {loading ? (
              <div className="grid gap-3 sm:grid-cols-4">
                {[0, 1, 2, 3].map((index) => <div key={index} className="h-24 animate-pulse rounded-2xl bg-white" />)}
              </div>
            ) : failed ? (
              <div className="card p-3"><StateBlock kind="error" onRetry={load} /></div>
            ) : data && tab === "overview" ? (
              <>
                <section className="card p-5">
                  <h2 className="section-title">应用信息</h2>
                  <div className="mt-3 grid gap-3 sm:grid-cols-4">
                    {[
                      { label: "应用", value: data.app.name },
                      { label: "版本", value: data.app.version },
                      { label: "环境", value: data.app.env },
                      { label: "Schema", value: data.app.schema_version },
                    ].map((item) => (
                      <div key={item.label} className="rounded-2xl bg-canvas p-3">
                        <p className="text-[11px] text-muted">{item.label}</p>
                        <p className="mt-1 break-all text-[13px] font-medium text-ink">{item.value || "—"}</p>
                      </div>
                    ))}
                  </div>
                </section>

                <section className="card p-5">
                  <h2 className="section-title">数据规模</h2>
                  <div className="mt-3 grid gap-3 sm:grid-cols-4">
                    {Object.entries(data.counts).map(([key, value]) => (
                      <div key={key} className="rounded-2xl bg-canvas p-3">
                        <p className="text-[11px] text-muted">{COUNT_LABEL[key] || key}</p>
                        <p className="mt-1 text-[22px] font-semibold text-ink">{value}</p>
                      </div>
                    ))}
                  </div>
                </section>
              </>
            ) : data && tab === "config" ? (
              <section className="card p-5">
                <h2 className="section-title">进程开关（只读）</h2>
                <ul className="mt-3 space-y-2 text-[13px]">
                  <li className="flex items-center justify-between rounded-2xl bg-canvas px-3 py-2">
                    <span>鉴权开关 <span className="text-[11px] text-muted">AUTH_ENABLED</span></span>
                    <span className={data.flags.auth_enabled ? "badge bg-emerald-50 text-emerald-700" : "badge bg-rose-50 text-rose-700"}>
                      {data.flags.auth_enabled ? "开启" : "关闭"}
                    </span>
                  </li>
                  <li className="flex items-center justify-between rounded-2xl bg-canvas px-3 py-2">
                    <span>医生签名 <span className="text-[11px] text-muted">CLINICIAN_WRITE_ENABLED</span></span>
                    <span className={data.flags.clinician_write_enabled ? "badge bg-emerald-50 text-emerald-700" : "badge bg-amber-50 text-amber-700"}>
                      {data.flags.clinician_write_enabled ? "已启用" : "未启用（仅草稿）"}
                    </span>
                  </li>
                  <li className="flex items-center justify-between rounded-2xl bg-canvas px-3 py-2">
                    <span>每分钟限流</span><span className="text-ink">{data.flags.rate_limit_per_minute} 次</span>
                  </li>
                  <li className="flex items-center justify-between rounded-2xl bg-canvas px-3 py-2">
                    <span>每日问诊配额</span><span className="text-ink">{data.flags.chat_daily_limit} 次</span>
                  </li>
                </ul>
                <p className="mt-3 text-[11px] leading-5 text-muted">
                  这里展示的是**进程级开关**（改环境变量后重启生效），所以本页只读。
                  业务侧配置（规则 / 通知模板 / Prompt / Agent 参数）已支持在线草稿 → 发布 → 回滚，
                  请到 <Link href="/platform/config/" className="text-primary">配置管理</Link> 操作，全程留审计。
                </p>
              </section>
            ) : tab === "compliance" && compliance ? (
              <div className="grid gap-4 lg:grid-cols-2">
                <section className="card p-5">
                  <h2 className="section-title">留存策略</h2>
                  <ul className="mt-3 space-y-2 text-[13px]">
                    {Object.entries(compliance.retention).map(([key, value]) => (
                      <li key={key} className="flex items-center justify-between rounded-2xl bg-canvas px-3 py-2">
                        <span className="text-muted">{key}</span><span className="text-ink">{String(value)}</span>
                      </li>
                    ))}
                  </ul>
                </section>
                <section className="card p-5">
                  <h2 className="section-title">脱敏规则</h2>
                  <ul className="mt-3 space-y-2 text-[13px] text-ink">
                    {compliance.redaction.map((item) => (
                      <li key={item} className="rounded-2xl bg-canvas px-3 py-2">{item}</li>
                    ))}
                  </ul>
                </section>
                <section className="card p-5">
                  <h2 className="section-title">数据接口</h2>
                  <ul className="mt-3 space-y-2 text-[13px]">
                    {Object.entries(compliance.endpoints).map(([key, value]) => (
                      <li key={key} className="flex items-center justify-between rounded-2xl bg-canvas px-3 py-2">
                        <span className="text-muted">{key}</span><code className="text-ink">{value}</code>
                      </li>
                    ))}
                  </ul>
                </section>
                <section className="card p-5">
                  <h2 className="section-title">数据规模</h2>
                  <div className="mt-3 grid grid-cols-2 gap-2">
                    {Object.entries(compliance.counts).map(([key, value]) => (
                      <div key={key} className="rounded-2xl bg-canvas p-3">
                        <p className="text-[11px] text-muted">{key}</p>
                        <p className="mt-1 text-[20px] font-semibold text-ink">{value}</p>
                      </div>
                    ))}
                  </div>
                </section>
              </div>
            ) : tab === "health" && health ? (
              <section className="card p-5">
                <h2 className="section-title">告警与 SLO（实时检查）</h2>
                <ul className="mt-3 space-y-2">
                  {health.checks.map((item) => (
                    <li key={item.name} className="flex flex-wrap items-center gap-2 rounded-2xl bg-canvas px-3 py-2 text-[13px]">
                      <span className={item.status === "ok" ? "badge bg-emerald-50 text-emerald-700" : item.status === "warn" ? "badge bg-amber-50 text-amber-700" : item.status === "fail" ? "badge bg-rose-50 text-rose-700" : "badge"}>
                        {item.status === "ok" ? "正常" : item.status === "warn" ? "关注" : item.status === "fail" ? "异常" : "信息"}
                      </span>
                      <span className="font-medium text-ink">{item.name}</span>
                      <span className="text-muted">{item.detail}</span>
                    </li>
                  ))}
                </ul>
                <p className="mt-3 text-[11px] leading-5 text-muted">
                  当前为规则实时检查（库连接、Schema、知识库、配置、审计、模型调用、评测批次、鉴权）。
                  通知渠道（钉钉 / 企业微信 / 电话）需要院方账号，接入后在「通知记录」里显示发送结果。
                </p>
              </section>
            ) : tab === "evaluation" && evaluation ? (
              <section className="card overflow-hidden">
                <div className="flex items-center justify-between px-4 py-3">
                  <h2 className="section-title">质量评测</h2>
                  <span className="text-[11px] text-muted">共 {evaluation.total} 次</span>
                </div>
                {evaluation.runs.length === 0 ? (
                  <div className="border-t border-line p-3">
                    <StateBlock kind="empty" title="还没有评测记录" description="运行评测后会在这里留下批次与指标" />
                  </div>
                ) : (
                  <table className="w-full text-left text-[12px]">
                    <thead className="bg-canvas text-muted">
                      <tr>
                        <th className="px-3 py-2 font-medium">批次</th>
                        <th className="px-3 py-2 font-medium">时间</th>
                        <th className="px-3 py-2 font-medium">指标</th>
                      </tr>
                    </thead>
                    <tbody>
                      {evaluation.runs.map((run) => (
                        <tr key={run.id} className="border-t border-line">
                          <td className="px-3 py-2 text-ink">{run.name}</td>
                          <td className="px-3 py-2 text-muted">{run.created_at?.slice(0, 19).replace("T", " ")}</td>
                          <td className="px-3 py-2 text-muted">
                            {Object.entries(run.metrics || {}).slice(0, 3).map(([key, value]) => key + "=" + String(value)).join(" · ") || "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </section>
            ) : data ? (
              <section className="card overflow-hidden">
                <div className="flex items-center justify-between px-4 py-3">
                  <h2 className="section-title">能力矩阵</h2>
                  <span className="text-[11px] text-muted">HIS 原生 / 平台自建 / 待确认</span>
                </div>
                <table className="w-full text-left text-[12px]">
                  <thead className="bg-canvas text-muted">
                    <tr>
                      <th className="px-3 py-2 font-medium">能力</th>
                      <th className="px-3 py-2 font-medium">归属</th>
                      <th className="px-3 py-2 font-medium">说明</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.capabilities.map((item) => (
                      <tr key={item.name} className="border-t border-line">
                        <td className="px-3 py-2 text-ink">{item.name}</td>
                        <td className="px-3 py-2">
                          <span className={item.owner === "HIS" ? "badge bg-primary-soft text-primary" : item.owner === "待确认" ? "badge bg-amber-50 text-amber-700" : "badge bg-violet-50 text-violet-700"}>
                            {item.owner}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-muted">{item.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            ) : null}
          </>
        )}

        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 px-1 text-[11px] text-muted">
          <Cpu size={13} />
          本页是运行状态视图：规则 / 模板 / Prompt 到
          <Link href="/platform/config/" className="text-primary">配置管理</Link>
          发布，索引重建在
          <Link href="/platform/knowledge/" className="text-primary">知识库</Link>
          ，灰度实验在
          <Link href="/platform/integrations/" className="text-primary">接入与实验</Link>。
          <span className="ml-2 inline-flex items-center gap-1" title="接口：/api/v1/admin/platform">
            <Database size={13} /> 数据来源：平台运行接口
          </span>
        </div>
      </main>
    </div>
  );
}
