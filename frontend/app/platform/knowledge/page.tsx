"use client";

import { useCallback, useEffect, useState } from "react";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { PlatformNav } from "@/components/PlatformNav";
import { statusOf, useAdminLoad } from "@/lib/use-admin-load";
import {
  debugRetrieval,
  getKbOverview,
  listRagDocuments,
  reindexRag,
  listUnmatched,
  resolveUnmatched,
  runPipelineCheck,
  type KbOverview,
  type PipelineCheck,
  type RagDocument,
  type RetrievalDebug,
  type UnmatchedItem,
} from "@/lib/api-admin-ext";

type Tab = "kb" | "rag" | "unmatched" | "debug" | "pipeline";

const TABS: Array<{ key: Tab; label: string; screen: string }> = [
  { key: "kb", label: "知识库策略", screen: "a-kb" },
  { key: "rag", label: "语料与索引", screen: "a-rag" },
  { key: "unmatched", label: "未命中问题", screen: "a-unmatched" },
  { key: "debug", label: "检索调试", screen: "a-debug" },
  { key: "pipeline", label: "RAG 全链路", screen: "a-ragflow" },
];

function formatTime(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { hour12: false });
}

export default function KnowledgePage() {
  const [tab, setTab] = useState<Tab>("kb");
  const [overview, setOverview] = useState<KbOverview | null>(null);
  const [docs, setDocs] = useState<RagDocument[]>([]);
  const [unmatched, setUnmatched] = useState<UnmatchedItem[]>([]);
  const [openOnly, setOpenOnly] = useState(true);
  const [query, setQuery] = useState("高血压患者平时要注意什么");
  const [department, setDepartment] = useState("");
  const [debug, setDebug] = useState<RetrievalDebug | null>(null);
  const [pipeline, setPipeline] = useState<PipelineCheck | null>(null);
  const [answer, setAnswer] = useState("");
  const [target, setTarget] = useState<UnmatchedItem | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [denied, setDenied] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setDenied(false);
    try {
      const [kb, rag, um] = await Promise.all([
        getKbOverview(),
        listRagDocuments(),
        listUnmatched(openOnly),
      ]);
      setOverview(kb);
      setDocs(rag.items);
      setUnmatched(um.items);
    } catch (err) {
      if (statusOf(err) === "denied") setDenied(true);
      else setError((err as Error).message || "加载失败");
    } finally {
      setLoading(false);
    }
  }, [openOnly]);

  useEffect(() => {
    void load();
  }, [load]);

  const submitResolve = async () => {
    if (!target) return;
    setError(null);
    try {
      await resolveUnmatched({
        question: target.question,
        answer,
        department: target.department,
        question_key: target.question_key,
      });
      setAnswer("");
      setTarget(null);
      await load();
    } catch (err) {
      setError((err as Error).message || "转知识条目失败");
    }
  };

  const runDebug = async () => {
    setError(null);
    try {
      setDebug(await debugRetrieval(query, department || undefined));
    } catch (err) {
      setError((err as Error).message || "检索失败");
    }
  };

  const runPipeline = async () => {
    setError(null);
    try {
      setPipeline(await runPipelineCheck(query, department || undefined));
    } catch (err) {
      setError((err as Error).message || "链路自检失败");
    }
  };

  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-6">
      <header className="flex flex-wrap items-center gap-3">
        <div>
          <h1 className="text-[20px] font-semibold text-ink">知识库与检索</h1>
          <p className="mt-1 text-[12px] text-muted">文档覆盖、未命中问题、检索调试与 RAG 链路自检</p>
        </div>
        <div className="ml-auto"><SourceBadge live /></div>
      </header>

      <div className="mt-3"><PlatformNav /></div>
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
      {message ? <p className="mt-3 rounded-2xl bg-primary-soft px-4 py-2 text-[12px] text-primary">{message}</p> : null}

      {loading ? (
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          {[0, 1, 2].map((i) => <div key={i} className="card h-28 animate-pulse bg-canvas" />)}
        </div>
      ) : denied ? (
        <div className="mt-4 card p-5"><StateBlock kind="denied" title="需要管理员权限" /></div>
      ) : (
        <div className="mt-4 space-y-4">
          {tab === "kb" && overview ? (
            <>
              <section className="card p-5">
                <h2 className="section-title">覆盖情况</h2>
                <div className="mt-3 grid gap-3 sm:grid-cols-4">
                  {[
                    { label: "文档总数", value: overview.documents },
                    { label: "科室覆盖", value: overview.coverage.covered + " / " + overview.coverage.total },
                    { label: "未命中问题", value: overview.unmatched },
                    { label: "已转知识", value: overview.resolved },
                  ].map((item) => (
                    <div key={item.label} className="rounded-2xl bg-canvas p-3">
                      <p className="text-[11px] text-muted">{item.label}</p>
                      <p className="mt-1 text-[20px] font-semibold text-ink">{item.value}</p>
                    </div>
                  ))}
                </div>
                {overview.coverage.missing.length ? (
                  <p className="mt-3 rounded-2xl bg-amber-50 px-3 py-2 text-[12px] text-amber-700">
                    未覆盖科室：{overview.coverage.missing.join("、")}
                  </p>
                ) : (
                  <p className="mt-3 text-[12px] text-emerald-700">21 个科室均有语料</p>
                )}
              </section>

              <section className="card p-5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h2 className="section-title">索引维护</h2>
                  <button
                    onClick={() =>
                      void (async () => {
                        setError(null);
                        setMessage(null);
                        try {
                          const result = await reindexRag();
                          setMessage(
                            "已重建全文索引：" + result.documents + " 条，耗时 " + result.ms + "ms（向量不重算）",
                          );
                          await load();
                        } catch (err) {
                          setError((err as Error).message || "重建失败");
                        }
                      })()
                    }
                    className="rounded-full bg-primary px-4 py-1.5 text-[12px] font-medium text-white"
                  >
                    重建全文索引
                  </button>
                </div>
                <p className="mt-2 text-[11px] leading-5 text-muted">
                  重建的是关键字检索用的 tsvector；向量在语料入库时由 embedding 模型写入，重跑代价高，不在本操作范围内。
                </p>
              </section>

              <section className="card p-5">
                <h2 className="section-title">检索参数（来自 Agent 配置 · 已发布值）</h2>
                <pre className="mt-3 overflow-auto rounded-2xl bg-canvas p-3 text-[12px] text-ink">
                  {JSON.stringify(overview.retrieval_config, null, 2)}
                </pre>
                <p className="mt-2 text-[11px] text-muted">修改请到「配置管理 → Agent 配置 → retrieval」。</p>
              </section>
            </>
          ) : null}

          {tab === "rag" ? (
            <section className="card overflow-hidden">
              <div className="flex items-center justify-between px-4 py-3">
                <h2 className="section-title">语料与索引</h2>
                <span className="text-[11px] text-muted">共 {docs.length} 条</span>
              </div>
              {docs.length === 0 ? (
                <div className="border-t border-line p-4"><StateBlock kind="empty" title="还没有语料" /></div>
              ) : (
                <ul className="divide-y divide-line border-t border-line">
                  {docs.map((doc) => (
                    <li key={doc.id} className="flex flex-wrap items-center gap-2 px-4 py-2 text-[12px]">
                      <span className="font-medium text-ink">{doc.title}</span>
                      <span className="badge bg-canvas text-muted">{doc.department || "未标注"}</span>
                      <span className="text-muted">{doc.source || "—"}</span>
                      <span className="ml-auto text-[11px] text-muted">{doc.chars} 字 · {formatTime(doc.created_at)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          ) : null}

          {tab === "unmatched" ? (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={() => setOpenOnly((prev) => !prev)}
                  className="rounded-full bg-white px-4 py-1.5 text-[12px] font-medium text-ink ring-1 ring-line"
                >
                  {openOnly ? "只看待处理" : "全部问题"}
                </button>
                <button onClick={() => void load()} className="rounded-full bg-primary px-4 py-1.5 text-[12px] font-medium text-white">
                  刷新
                </button>
              </div>
              <section className="card overflow-hidden">
                {unmatched.length === 0 ? (
                  <div className="p-4">
                    <StateBlock kind="empty" title="没有未命中问题" description="检索每次都命中了知识库；很好" />
                  </div>
                ) : (
                  <ul className="divide-y divide-line">
                    {unmatched.map((item) => (
                      <li key={item.question_key} className="px-4 py-3">
                        <div className="flex flex-wrap items-center gap-2 text-[13px]">
                          <span className="font-medium text-ink">{item.question}</span>
                          <span className="badge bg-canvas text-muted">{item.department || "未标科室"}</span>
                          <span className="text-[11px] text-muted">命中 {item.hits} 次</span>
                          {item.resolved ? (
                            <span className="badge bg-emerald-50 text-emerald-700">已转 {item.resolved_doc_key}</span>
                          ) : null}
                          <span className="ml-auto text-[11px] text-muted">{formatTime(item.last_seen)}</span>
                          {!item.resolved ? (
                            <button
                              onClick={() => { setTarget(item); setAnswer(""); }}
                              className="text-[12px] font-medium text-primary"
                            >
                              转知识条目
                            </button>
                          ) : null}
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </section>

              {target ? (
                <section className="card p-4">
                  <h2 className="section-title">把「{target.question}」转成知识条目</h2>
                  <textarea
                    value={answer}
                    onChange={(event) => setAnswer(event.target.value)}
                    aria-label="知识条目答案"
                    rows={4}
                    placeholder="写清标准答案与出处；发布后会进入知识库并参与检索"
                    className="mt-2 w-full rounded-2xl bg-canvas p-3 text-[13px] text-ink"
                  />
                  <div className="mt-2 flex gap-2">
                    <button
                      onClick={() => void submitResolve()}
                      className="rounded-full bg-primary px-4 py-1.5 text-[12px] font-medium text-white"
                    >
                      写入知识库
                    </button>
                    <button onClick={() => setTarget(null)} className="rounded-full bg-white px-4 py-1.5 text-[12px] text-muted ring-1 ring-line">
                      取消
                    </button>
                  </div>
                </section>
              ) : null}
            </>
          ) : null}

          {tab === "debug" ? (
            <>
              <section className="card p-4">
                <div className="flex flex-wrap items-end gap-2">
                  <label className="flex-1">
                    <span className="text-[11px] text-muted">问题</span>
                    <input
                      value={query}
                      onChange={(event) => setQuery(event.target.value)}
                      className="mt-1 w-full rounded-xl bg-canvas px-3 py-2 text-[13px] text-ink"
                    />
                  </label>
                  <label>
                    <span className="text-[11px] text-muted">科室（可选）</span>
                    <input
                      value={department}
                      onChange={(event) => setDepartment(event.target.value)}
                      placeholder="如 心内科"
                      className="mt-1 rounded-xl bg-canvas px-3 py-2 text-[13px] text-ink"
                    />
                  </label>
                  <button onClick={() => void runDebug()} className="rounded-full bg-primary px-4 py-2 text-[12px] font-medium text-white">
                    检索
                  </button>
                </div>
                {debug ? (
                  <p className="mt-3 text-[12px] text-muted">
                    命中 {debug.total} 条 · {debug.latency_ms}ms · 科室过滤 {debug.filtered ? "生效" : "未生效"} ·
                    科室无数据回退 {debug.dept_unavailable ? "是" : "否"} · top_k {debug.settings.top_k}
                  </p>
                ) : null}
              </section>

              {debug && debug.hits.length > 0 ? (
                <section className="card overflow-hidden">
                  <ul className="divide-y divide-line">
                    {debug.hits.map((hit) => (
                      <li key={hit.doc_key} className="px-4 py-3">
                        <div className="flex items-center gap-2 text-[13px]">
                          <span className="font-medium text-ink">{hit.title}</span>
                          <span className="badge bg-canvas text-muted">{hit.department || "未标注"}</span>
                          <span className="ml-auto text-[11px] text-primary">score {hit.score}</span>
                        </div>
                        <p className="mt-1 text-[12px] leading-5 text-muted">{hit.snippet}</p>
                      </li>
                    ))}
                  </ul>
                </section>
              ) : debug ? (
                <section className="card p-4">
                  <StateBlock kind="empty" title="没有命中任何知识" description="该问题会进入未命中问题池，可在「未命中问题」里转成知识条目" />
                </section>
              ) : null}
            </>
          ) : null}

          {tab === "pipeline" ? (
            <>
              <section className="card p-4">
                <div className="flex flex-wrap items-end gap-2">
                  <label className="flex-1">
                    <span className="text-[11px] text-muted">用一条真实问题跑链路自检</span>
                    <input
                      value={query}
                      onChange={(event) => setQuery(event.target.value)}
                      className="mt-1 w-full rounded-xl bg-canvas px-3 py-2 text-[13px] text-ink"
                    />
                  </label>
                  <button onClick={() => void runPipeline()} className="rounded-full bg-primary px-4 py-2 text-[12px] font-medium text-white">
                    跑一遍
                  </button>
                </div>
              </section>

              {pipeline ? (
                <>
                  <section className="card p-4">
                    <div className="flex items-center justify-between">
                      <h2 className="section-title">链路结果</h2>
                      <span className="text-[11px] text-muted">
                        命中 {pipeline.hits} · {pipeline.latency_ms}ms
                      </span>
                    </div>
                    {pipeline.error ? (
                      <p className="mt-2 rounded-2xl bg-rose-50 px-3 py-2 text-[12px] text-rose-700">{pipeline.error}</p>
                    ) : null}
                    <ul className="mt-3 space-y-2">
                      {pipeline.stages.map((stage) => (
                        <li key={stage.name} className="flex flex-wrap items-center gap-2 rounded-2xl bg-canvas px-3 py-2 text-[12px]">
                          <span className={stage.ok ? "badge bg-emerald-50 text-emerald-700" : "badge bg-rose-50 text-rose-700"}>
                            {stage.ok ? "通过" : "失败"}
                          </span>
                          <span className="font-medium text-ink">{stage.name}</span>
                          <span className="text-muted">{stage.detail}</span>
                        </li>
                      ))}
                    </ul>
                  </section>
                  {pipeline.top.length ? (
                    <section className="card p-4">
                      <h2 className="section-title">Top 证据</h2>
                      <ul className="mt-2 space-y-1 text-[12px]">
                        {pipeline.top.map((hit) => (
                          <li key={hit.doc_key} className="flex items-center gap-2">
                            <span className="text-primary">{hit.score}</span>
                            <span className="text-ink">{hit.title}</span>
                            <span className="text-muted">{hit.department}</span>
                          </li>
                        ))}
                      </ul>
                    </section>
                  ) : null}
                </>
              ) : null}
            </>
          ) : null}
        </div>
      )}
    </main>
  );
}
