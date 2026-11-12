"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Activity, ArrowRight, CircleDot } from "lucide-react";
import { getAgentPanel, type AgentPanelData } from "@/lib/api";

/** 右栏 Agent 面板：能力、最近调用、边界。数据全部来自 /api/v1/agent/panel。 */
const LABEL: Record<AgentPanelData["end"], string> = {
  patient: "患者端 Agent",
  clinician: "医生端 Agent",
  ops: "运营 Agent",
  platform: "平台 Agent",
};

export function AgentPanel({
  end,
  className = "",
}: {
  end?: "patient" | "clinician" | "ops" | "platform";
  className?: string;
}) {
  const [data, setData] = useState<AgentPanelData | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    getAgentPanel(end)
      .then((value) => {
        if (alive) setData(value);
      })
      .catch(() => {
        if (alive) setFailed(true);
      });
    return () => {
      alive = false;
    };
  }, [end]);

  return (
    <aside
      className={"agent-panel " + className}
      aria-label="Agent 运行时面板"
      data-testid="agent-panel"
    >
      <div className="agent-panel-head">
        <span className="inline-flex items-center gap-2 text-[13px] font-semibold text-ink">
          <Activity size={14} className="text-primary" />
          {data ? LABEL[data.end] : "Agent"}
        </span>
        <span className="text-[11px] text-muted">{data ? (data.end === "patient" ? "解释与事务代办" : data.end === "platform" ? "运维助手 · 只读建议" : "随行 · 领域约束") : "…"}</span>
      </div>

      {failed ? (
        <p className="px-4 py-3 text-[11px] leading-5 text-muted">面板暂不可用（接口未响应），不影响主流程。</p>
      ) : null}

      <div className="agent-panel-body">
        <section>
          <p className="agent-panel-title">可用能力</p>
          <div className="flex flex-wrap gap-1.5">
            {(data?.capabilities || []).map((cap) => (
              <span key={cap.name} className="agent-chip" title={cap.note}>
                {cap.name}
              </span>
            ))}
          </div>
        </section>

        <section>
          <p className="agent-panel-title">
            最近调用
            {data ? <span className="ml-2 font-normal text-muted">{data.scope === "tenant" ? "本院区" : "本账号"} · 7 天 {data.summary.calls_7d} 次 · 均值 {data.summary.avg_latency_ms}ms</span> : null}
          </p>
          {data && data.recent.length === 0 ? (
            <p className="text-[11px] leading-5 text-muted">近 7 天没有模型调用记录。</p>
          ) : null}
          <ul className="space-y-1.5">
            {(data?.recent || []).map((row, index) => (
              <li key={row.name + index} className="flex items-center gap-2 rounded-lg bg-white px-2 py-1.5 text-[11px]">
                <CircleDot size={10} className="shrink-0 text-emerald-500" />
                <span className="min-w-0 flex-1 truncate text-ink">{row.name}</span>
                <span className="shrink-0 text-muted">{row.latency_ms}ms</span>
              </li>
            ))}
          </ul>
        </section>

        <section>
          <p className="agent-panel-title">边界</p>
          <ul className="space-y-1 text-[11px] leading-5 text-muted">
            {(data?.boundaries || []).map((line) => (
              <li key={line}>· {line}</li>
            ))}
          </ul>
        </section>

        <section className="mt-auto rounded-xl bg-primary-soft/60 p-3">
          <p className="text-[11px] leading-5 text-primary">{data?.starter || "问：今天有什么需要注意的？"}</p>
          <Link
            href={data?.end === "patient" ? "/consult/" : data?.end === "clinician" ? "/clinician/" : data?.end === "ops" ? "/ops/" : "/platform/"}
            className="mt-2 inline-flex items-center gap-1 text-[11px] font-medium text-primary"
          >
            进入对话 <ArrowRight size={12} />
          </Link>
        </section>
      </div>
    </aside>
  );
}
