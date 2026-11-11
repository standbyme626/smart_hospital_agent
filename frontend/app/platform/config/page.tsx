"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Plus, RefreshCw, Save, Undo2, Upload } from "lucide-react";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { ApiError } from "@/lib/api";
import {
  getConfigHistory,
  getConfigKinds,
  listConfigEntries,
  previewNotifyTemplate,
  publishConfig,
  rollbackConfig,
  saveConfigDraft,
  setConfigEnabled,
  type ConfigEntry,
  type ConfigHistoryItem,
  type ConfigKind,
  type ConfigKindInfo,
  type NotifyPreview,
} from "@/lib/api-admin-config";

type LoadState = "loading" | "ready" | "denied" | "error" | "timeout";

const SCAFFOLD: Record<ConfigKind, Record<string, unknown>> = {
  rule: { severity: "high", keywords: [], action: "命中后提示咨询医生" },
  notify_template: {
    title: "新模板",
    body: "{{member}}你好，{{content}}",
    channels: ["inbox"],
    retry: { max: 3, backoff_s: 30 },
  },
  prompt: { label: "新提示词", content: "你是……", model: "default", temperature: 0.2 },
  agent_config: { note: "新配置项" },
};

const KIND_HINT: Record<ConfigKind, string> = {
  rule: "危机识别 / DDI / 危急值等确定性规则；发布后立即生效，停用即回退到「未启用」。",
  notify_template: "站内信与未来短信 / 微信共用模板；支持 {{变量}} 预览，未提供的变量会在预览里列出。",
  prompt: "Agent 各环节提示词与模型参数；草稿可反复修改，发布才改变线上行为。",
  agent_config: "模型路由、检索参数与安全护栏；发布前避免配置成付费模型。",
};

function statusOf(error: unknown): LoadState {
  const status = (error as { status?: number } | null)?.status ?? 0;
  if (status === 401 || status === 403) return "denied";
  if (status === 408 || status === 504) return "timeout";
  return "error";
}

function formatTime(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-CN", { hour12: false });
}

export default function PlatformConfigPage() {
  const [kinds, setKinds] = useState<ConfigKindInfo[]>([]);
  const [kind, setKind] = useState<ConfigKind>("rule");
  const [entries, setEntries] = useState<ConfigEntry[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [draftText, setDraftText] = useState("");
  const [history, setHistory] = useState<ConfigHistoryItem[]>([]);
  const [state, setState] = useState<LoadState>("loading");
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  const [preview, setPreview] = useState<NotifyPreview | null>(null);
  const [previewVars, setPreviewVars] = useState('{"member":"张三","kind":"血常规","due":"11-20"}');
  const [newKey, setNewKey] = useState("");

  const entry = useMemo(() => entries.find((item) => item.key === selected) ?? null, [entries, selected]);

  const loadEntries = useCallback(async (target: ConfigKind) => {
    setState("loading");
    try {
      const items = await listConfigEntries(target);
      setEntries(items);
      setSelected((prev) => (prev && items.some((item) => item.key === prev) ? prev : items[0]?.key ?? null));
      setState("ready");
    } catch (error) {
      setState(statusOf(error));
    }
  }, []);

  useEffect(() => {
    let alive = true;
    getConfigKinds()
      .then((items) => {
        if (!alive) return;
        setKinds(items);
        const first = (items[0]?.kind ?? "rule") as ConfigKind;
        setKind(first);
        void loadEntries(first);
      })
      .catch((error) => {
        if (alive) setState(statusOf(error));
      });
    return () => {
      alive = false;
    };
  }, [loadEntries]);

  useEffect(() => {
    if (entry) setDraftText(JSON.stringify(entry.draft, null, 2));
    setPreview(null);
    if (!selected || state !== "ready") {
      setHistory([]);
      return;
    }
    let alive = true;
    getConfigHistory(kind, selected)
      .then((items) => {
        if (alive) setHistory(items);
      })
      .catch(() => {
        if (alive) setHistory([]);
      });
    return () => {
      alive = false;
    };
  }, [entry, kind, selected, state]);

  const switchKind = (next: ConfigKind) => {
    setKind(next);
    setSelected(null);
    setDraftText("");
    setPreview(null);
    void loadEntries(next);
  };

  const parseDraft = (): Record<string, unknown> | null => {
    try {
      const parsed = JSON.parse(draftText) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        setMessage("配置内容必须是 JSON 对象，例如 {\"severity\":\"high\"}");
        return null;
      }
      return parsed as Record<string, unknown>;
    } catch (error) {
      setMessage("JSON 格式有误：" + (error as Error).message);
      return null;
    }
  };

  const run = async (action: () => Promise<void>) => {
    setBusy(true);
    setMessage(null);
    try {
      await action();
    } catch (error) {
      const next = statusOf(error);
      if (next === "denied" || next === "timeout") setState(next);
      setMessage((error as Error).message || "操作失败");
    } finally {
      setBusy(false);
    }
  };

  const handleSave = () =>
    run(async () => {
      const value = parseDraft();
      if (!value) return;
      const key = selected && entry ? selected : newKey.trim();
      if (!key) {
        setMessage("请先选择条目或填写新建 key");
        return;
      }
      const saved = await saveConfigDraft(kind, key, value, note || undefined);
      setSelected(saved.key);
      await loadEntries(kind);
      setMessage("草稿已保存 v" + saved.version + "（尚未发布）");
    });

  const handlePublish = () =>
    run(async () => {
      if (!selected) return;
      const published = await publishConfig(kind, selected, note || undefined);
      await loadEntries(kind);
      setMessage("已发布 v" + published.version + "，线上生效值已更新");
    });

  const handleRollback = (version: number) =>
    run(async () => {
      if (!selected) return;
      const rolled = await rollbackConfig(kind, selected, version);
      await loadEntries(kind);
      setMessage("已回滚到 v" + version + "，新版本号 v" + rolled.version);
    });

  const handleToggle = () =>
    run(async () => {
      if (!entry) return;
      const next = await setConfigEnabled(kind, entry.key, !entry.enabled);
      await loadEntries(kind);
      setMessage(next.enabled ? "已启用" : "已停用：页面显示「未启用」，不沿用旧值");
    });

  const handlePreview = () =>
    run(async () => {
      if (!selected) return;
      let vars: Record<string, string> = {};
      try {
        vars = JSON.parse(previewVars || "{}") as Record<string, string>;
      } catch (error) {
        setMessage("变量必须是 JSON 对象：" + (error as Error).message);
        return;
      }
      setPreview(await previewNotifyTemplate(selected, vars));
    });

  const startNew = () => {
    const key = newKey.trim();
    if (!/^[a-z0-9_]{2,40}$/.test(key)) {
      setMessage("key 只允许小写字母、数字与下划线（2-40 位）");
      return;
    }
    setSelected(null);
    setDraftText(JSON.stringify(SCAFFOLD[kind], null, 2));
    setMessage("已填入模板，填好后点「保存草稿」即创建 {key} 的草稿".replace("{key}", key));
  };

  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <Link href="/platform" className="inline-flex items-center gap-1 text-[13px] text-muted">
            <ArrowLeft size={14} /> 返回平台
          </Link>
          <h1 className="text-[20px] font-semibold text-ink">配置管理</h1>
          <SourceBadge live />
        </div>
        <button
          onClick={() => void loadEntries(kind)}
          className="inline-flex min-h-9 items-center gap-1 rounded-full bg-white px-4 text-[12px] font-medium text-primary"
        >
          <RefreshCw size={13} /> 刷新
        </button>
      </div>

      <div className="mt-3">
        <OfflineBanner />
      </div>

      <nav className="mt-4 flex flex-wrap gap-2">
        {(kinds.length ? kinds : [{ kind: kind, name: "加载中", total: 0, published: 0, disabled: 0, description: "" }]).map((item) => (
          <button
            key={item.kind}
            onClick={() => switchKind(item.kind as ConfigKind)}
            className={
              item.kind === kind
                ? "rounded-full bg-primary px-4 py-1.5 text-[12px] font-medium text-white"
                : "rounded-full bg-white px-4 py-1.5 text-[12px] font-medium text-ink"
            }
          >
            {item.name}
            <span className="ml-1 text-[11px] opacity-70">{item.total}</span>
          </button>
        ))}
      </nav>

      <p className="mt-3 text-[12px] leading-5 text-muted">{KIND_HINT[kind]}</p>

      {message ? (
        <p className="mt-3 rounded-2xl bg-primary-soft px-4 py-2 text-[12px] text-primary">{message}</p>
      ) : null}

      {state === "loading" ? (
        <div className="mt-4 grid gap-3 lg:grid-cols-3">
          {[0, 1, 2].map((index) => (
            <div key={index} className="card h-40 animate-pulse bg-canvas" />
          ))}
        </div>
      ) : state === "denied" ? (
        <div className="mt-4 card p-5">
          <StateBlock kind="denied" title="需要管理员权限" description="配置管理只有 admin 角色可以访问" />
        </div>
      ) : state === "timeout" ? (
        <div className="mt-4 card p-5">
          <StateBlock kind="timeout" onRetry={() => void loadEntries(kind)} />
        </div>
      ) : state === "error" ? (
        <div className="mt-4 card p-5">
          <StateBlock kind="error" onRetry={() => void loadEntries(kind)} />
        </div>
      ) : (
        <div className="mt-4 grid gap-4 lg:grid-cols-[240px,1fr,300px]">
          <section className="card overflow-hidden">
            <div className="flex items-center justify-between px-3 py-2">
              <h2 className="text-[12px] font-medium text-ink">条目（{entries.length}）</h2>
            </div>
            {entries.length === 0 ? (
              <div className="border-t border-line p-3">
                <StateBlock kind="empty" title="还没有配置项" description="在右侧新建一个" />
              </div>
            ) : (
              <ul className="border-t border-line">
                {entries.map((item) => (
                  <li key={item.key}>
                    <button
                      onClick={() => setSelected(item.key)}
                      className={
                        item.key === selected
                          ? "flex w-full items-center justify-between gap-2 bg-primary-soft px-3 py-2 text-left text-[12px] text-primary"
                          : "flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-[12px] text-ink"
                      }
                    >
                      <span className="truncate">{item.key}</span>
                      <span className="flex shrink-0 items-center gap-1">
                        {item.dirty ? <span className="badge bg-amber-50 text-amber-700">草稿</span> : null}
                        {!item.enabled ? <span className="badge bg-rose-50 text-rose-700">停用</span> : null}
                        <span className="text-[11px] text-muted">v{item.version}</span>
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
            <div className="border-t border-line p-3">
              <label className="text-[11px] text-muted" htmlFor="new-key">新建条目 key</label>
              <div className="mt-1 flex gap-2">
                <input
                  id="new-key"
                  value={newKey}
                  onChange={(event) => setNewKey(event.target.value)}
                  placeholder="例如 payment_timeout"
                  className="min-h-9 w-full rounded-xl bg-canvas px-3 text-[12px] text-ink"
                />
                <button
                  onClick={startNew}
                  className="inline-flex min-h-9 shrink-0 items-center gap-1 rounded-xl bg-white px-3 text-[12px] font-medium text-primary"
                >
                  <Plus size={13} /> 新建
                </button>
              </div>
            </div>
          </section>

          <section className="card p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-[13px] font-medium text-ink">
                {entry ? entry.key : selected ? selected : "新建条目"}
                {entry ? (
                  <span className="ml-2 text-[11px] text-muted">
                    v{entry.version} · 生效 {entry.effective ? "已发布" : "未发布"}
                  </span>
                ) : null}
              </h2>
              <span className="text-[11px] text-muted">更新 {formatTime(entry?.updated_at ?? null)}</span>
            </div>

            {entry?.effective ? (
              <div className="mt-3 rounded-2xl bg-canvas p-3">
                <p className="text-[11px] text-muted">当前生效值</p>
                <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap break-all text-[11px] text-ink">
                  {JSON.stringify(entry.effective, null, 2)}
                </pre>
              </div>
            ) : null}

            <label className="mt-3 block text-[11px] text-muted" htmlFor="draft">草稿（JSON）</label>
            <textarea
              id="draft"
              value={draftText}
              onChange={(event) => setDraftText(event.target.value)}
              aria-label="草稿（JSON）"
              spellCheck={false}
              rows={14}
              className="mt-1 w-full rounded-2xl bg-canvas p-3 font-mono text-[12px] leading-5 text-ink"
            />
            <input
              value={note}
              onChange={(event) => setNote(event.target.value)}
              placeholder="变更说明（写入审计）"
              className="mt-2 min-h-9 w-full rounded-xl bg-canvas px-3 text-[12px] text-ink"
            />

            <div className="mt-3 flex flex-wrap gap-2">
              <button
                onClick={() => void handleSave()}
                disabled={busy}
                className="inline-flex min-h-9 items-center gap-1 rounded-full bg-white px-4 text-[12px] font-medium text-primary disabled:opacity-50"
              >
                <Save size={13} /> 保存草稿
              </button>
              <button
                onClick={() => void handlePublish()}
                disabled={busy || !entry}
                className="inline-flex min-h-9 items-center gap-1 rounded-full bg-primary px-4 text-[12px] font-medium text-white disabled:opacity-50"
              >
                <Upload size={13} /> 发布
              </button>
              <button
                onClick={() => void handleToggle()}
                disabled={busy || !entry}
                className="inline-flex min-h-9 items-center gap-1 rounded-full bg-white px-4 text-[12px] font-medium text-ink disabled:opacity-50"
              >
                {entry?.enabled ? "停用" : "启用"}
              </button>
            </div>

            {kind === "notify_template" && selected ? (
              <div className="mt-4 rounded-2xl bg-canvas p-3">
                <p className="text-[11px] text-muted">模板预览变量（JSON）</p>
                <textarea
                  value={previewVars}
                  onChange={(event) => setPreviewVars(event.target.value)}
                  aria-label="模板预览变量（JSON）"
                  rows={3}
                  spellCheck={false}
                  className="mt-1 w-full rounded-xl bg-white p-2 font-mono text-[11px] text-ink"
                />
                <button
                  onClick={() => void handlePreview()}
                  disabled={busy}
                  className="mt-2 inline-flex min-h-8 items-center rounded-full bg-white px-3 text-[12px] font-medium text-primary disabled:opacity-50"
                >
                  预览渲染结果
                </button>
                {preview ? (
                  <div className="mt-2 rounded-xl bg-white p-3 text-[12px] leading-5 text-ink">
                    <p className="font-medium">{preview.title || "（无标题）"}</p>
                    <p className="mt-1 whitespace-pre-wrap">{preview.body}</p>
                    <p className="mt-2 text-[11px] text-muted">
                      渠道：{preview.channels.join(" / ") || "—"} · 来源：
                      {preview.source === "published" ? "已发布" : "草稿"}
                    </p>
                    {preview.missing_vars.length ? (
                      <p className="mt-1 text-[11px] text-amber-700">
                        缺少变量：{preview.missing_vars.join("、")}
                      </p>
                    ) : null}
                  </div>
                ) : null}
              </div>
            ) : null}
          </section>

          <section className="card p-4">
            <h2 className="text-[13px] font-medium text-ink">版本历史</h2>
            {history.length === 0 ? (
              <div className="mt-3">
                <StateBlock kind="empty" title="还没有发布记录" description="发布或回滚后会留下版本时间线" />
              </div>
            ) : (
              <ul className="mt-3 space-y-2">
                {history.map((item) => (
                  <li key={item.version + "-" + item.created_at} className="rounded-2xl bg-canvas p-3">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[12px] font-medium text-ink">
                        v{item.version}
                        <span className={item.action === "rollback" ? "ml-2 badge bg-amber-50 text-amber-700" : "ml-2 badge bg-emerald-50 text-emerald-700"}>
                          {item.action === "rollback" ? "回滚" : "发布"}
                        </span>
                      </span>
                      <button
                        onClick={() => void handleRollback(item.version)}
                        disabled={busy}
                        className="inline-flex items-center gap-1 text-[11px] font-medium text-primary disabled:opacity-50"
                      >
                        <Undo2 size={12} /> 回滚到此版
                      </button>
                    </div>
                    <p className="mt-1 text-[11px] text-muted">
                      {formatTime(item.created_at)} · {item.actor || "—"}
                      {item.note ? " · " + item.note : ""}
                    </p>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      )}
    </main>
  );
}
