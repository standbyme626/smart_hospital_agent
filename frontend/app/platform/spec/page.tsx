"use client";

import { useCallback, useEffect, useState } from "react";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { ConsoleShell } from "@/components/ConsoleShell";
import { statusOf } from "@/lib/use-admin-load";
import { getFlow, getStates, type FlowData, type StatesData } from "@/lib/api-admin-ext";

type Tab = "flow" | "states";

const TABS: Array<{ key: Tab; label: string; screen: string }> = [
  { key: "flow", label: "数据流", screen: "a-flow" },
  { key: "states", label: "状态规范", screen: "a-states" },
];

const FLOW_NOTES: Record<string, string> = {
  知识库文档: "检索的语料来源；管理端可在「知识库 → 语料与索引」查看",
  问诊会话: "一次问诊 = 一个 session；患者端与消息中心都挂在它上面",
  用户提问: "进入检索与模型的原始输入",
  助手回答: "带证据引用；无证据时必须回答「未覆盖」",
  报告解读: "上传报告后的结构化结果与危急值",
  模型调用: "每次调用都写 model_calls，管理端「模型网关」按周期聚合",
  审计记录: "写操作与关键读操作的留痕，配置变更另存 config_history",
  未命中问题: "检索 0 命中时写入，可一键转成知识条目",
};

export default function SpecPage() {
  const [tab, setTab] = useState<Tab>("flow");
  const [flow, setFlow] = useState<FlowData | null>(null);
  const [states, setStates] = useState<StatesData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [denied, setDenied] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setDenied(false);
    try {
      const [flowData, stateData] = await Promise.all([getFlow(), getStates()]);
      setFlow(flowData);
      setStates(stateData);
    } catch (err) {
      if (statusOf(err) === "denied") setDenied(true);
      else setError((err as Error).message || "加载失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const maxCount = flow ? Math.max(1, ...flow.stages.map((stage) => stage.count)) : 1;

  return (
    <ConsoleShell end="platform" title="规范与数据流" subtitle="内部工具页：真实数据在各环节的流量，以及各状态机的实际取值">

      <div className="mt-3"><OfflineBanner /></div>

      <nav className="mt-4 flex flex-wrap gap-2">
        {TABS.map((item) => (
          <button
            key={item.key}
            onClick={() => setTab(item.key)}
            className={
              tab === item.key
                ? "rounded-full bg-primary px-4 py-1.5 text-[12px] font-medium text-white"
                : "rounded-full bg-white px-4 py-1.5 text-[12px] font-medium text-ink ring-1 ring-line"
            }
          >
            {item.label}
            <span className="ml-1 text-[10px] opacity-60">{item.screen}</span>
          </button>
        ))}
      </nav>

      {error ? <p className="mt-3 rounded-2xl bg-rose-50 px-4 py-2 text-[12px] text-rose-700">{error}</p> : null}

      {loading ? (
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          {[0, 1, 2].map((i) => <div key={i} className="card h-28 animate-pulse bg-canvas" />)}
        </div>
      ) : denied ? (
        <div className="mt-4 card p-5"><StateBlock kind="denied" title="需要管理员权限" /></div>
      ) : (
        <div className="mt-4 space-y-4">
          {tab === "flow" && flow ? (
            <section className="card p-5">
              <h2 className="section-title">数据流（实时计数）</h2>
              <ul className="mt-3 space-y-2">
                {flow.stages.map((stage) => (
                  <li key={stage.name} className="rounded-2xl bg-canvas px-3 py-2">
                    <div className="flex items-center justify-between text-[12px]">
                      <span className="font-medium text-ink">{stage.name}</span>
                      <span className="text-ink">{stage.count}</span>
                    </div>
                    <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-white">
                      <div
                        className="h-full rounded-full bg-primary"
                        style={{ width: Math.max(2, Math.round((stage.count / maxCount) * 100)) + "%" }}
                      />
                    </div>
                    <p className="mt-1 text-[11px] text-muted">{FLOW_NOTES[stage.name] || "—"}</p>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}

          {tab === "states" && states ? (
            <section className="card p-5">
              <h2 className="section-title">状态机实际取值</h2>
              <p className="mt-1 text-[11px] text-muted">数值取自线上库，未经处理；为空说明该状态机还没有数据。</p>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                {states.items.map((item) => (
                  <div key={item.label} className="rounded-2xl bg-canvas p-3">
                    <div className="flex items-center justify-between">
                      <span className="text-[12px] font-medium text-ink">{item.label}</span>
                      <span className="font-mono text-[10px] text-muted">{item.table}.{item.column}</span>
                    </div>
                    {item.error ? (
                      <p className="mt-1 text-[11px] text-rose-700">{item.error}</p>
                    ) : item.values.length === 0 ? (
                      <p className="mt-1 text-[11px] text-muted">暂无数据</p>
                    ) : (
                      <ul className="mt-2 space-y-1">
                        {item.values.map((value) => (
                          <li key={value.value} className="flex items-center justify-between text-[12px]">
                            <span className="text-muted">{value.value}</span>
                            <span className="text-ink">{value.count}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </div>
            </section>
          ) : null}
        </div>
      )}
    </ConsoleShell>
  );
}
