"use client";

import { useCallback, useEffect, useState } from "react";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { PlatformNav } from "@/components/PlatformNav";
import { statusOf } from "@/lib/use-admin-load";
import {
  createExperiment,
  createTenant,
  getHisOverview,
  listExperiments,
  listTenants,
  runHisCheck,
  updateExperiment,
  updateTenant,
  type ExperimentItem,
  type HisOverview,
  type TenantItem,
} from "@/lib/api-admin-ext";

type Tab = "his" | "tenant" | "experiment";

const TABS: Array<{ key: Tab; label: string; screen: string }> = [
  { key: "his", label: "HIS 接入", screen: "a-his" },
  { key: "tenant", label: "多院区", screen: "a-tenant" },
  { key: "experiment", label: "实验与灰度", screen: "a-experiment" },
];

const STATUS_LABEL: Record<string, string> = {
  active: "启用",
  suspended: "已暂停",
  archived: "已归档",
  draft: "草稿",
  running: "运行中",
  paused: "暂停",
};

function formatTime(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { hour12: false });
}

export default function IntegrationsPage() {
  const [tab, setTab] = useState<Tab>("his");
  const [his, setHis] = useState<HisOverview | null>(null);
  const [tenants, setTenants] = useState<TenantItem[]>([]);
  const [experiments, setExperiments] = useState<ExperimentItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [denied, setDenied] = useState(false);
  const [busy, setBusy] = useState(false);

  const [tenantCode, setTenantCode] = useState("");
  const [tenantName, setTenantName] = useState("");
  const [tenantIsolation, setTenantIsolation] = useState("logical");

  const [expKey, setExpKey] = useState("");
  const [expName, setExpName] = useState("");
  const [expVariants, setExpVariants] = useState("control,v2");

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setDenied(false);
    try {
      const [hisData, tenantData, expData] = await Promise.all([
        getHisOverview(),
        listTenants(),
        listExperiments(),
      ]);
      setHis(hisData);
      setTenants(tenantData.items);
      setExperiments(expData.items);
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

  const guard = async (action: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await action();
    } catch (err) {
      setError((err as Error).message || "操作失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-6">
      <header className="flex flex-wrap items-center gap-3">
        <div>
          <h1 className="text-[20px] font-semibold text-ink">接入与实验</h1>
          <p className="mt-1 text-[12px] text-muted">HIS 连通性、多院区与灰度实验（全部落库，变更留痕）</p>
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
      {notice ? <p className="mt-3 rounded-2xl bg-primary-soft px-4 py-2 text-[12px] text-primary">{notice}</p> : null}

      {loading ? (
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          {[0, 1, 2].map((i) => <div key={i} className="card h-28 animate-pulse bg-canvas" />)}
        </div>
      ) : denied ? (
        <div className="mt-4 card p-5"><StateBlock kind="denied" title="需要管理员权限" /></div>
      ) : (
        <div className="mt-4 space-y-4">
          {tab === "his" && his ? (
            <>
              <section className="card p-5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h2 className="section-title">HIS 连接配置（只读）</h2>
                  <button
                    onClick={() =>
                      void guard(async () => {
                        const result = await runHisCheck();
                        setNotice(
                          (result.ok ? "连接正常" : "连接异常") + "：" + result.detail + "（" + result.latency_ms + "ms）",
                        );
                        await load();
                      })
                    }
                    disabled={busy}
                    className="rounded-full bg-primary px-4 py-1.5 text-[12px] font-medium text-white disabled:opacity-50"
                  >
                    测试连接
                  </button>
                </div>
                <dl className="mt-3 grid gap-3 sm:grid-cols-2">
                  {[
                    ["MCP 地址", his.config.mcp_url || "未配置"],
                    ["支付模式", his.config.payment_mode || "—"],
                    ["医生签名写入", his.config.clinician_write_enabled ? "已启用" : "未启用（仅草稿）"],
                    ["鉴权", his.config.auth_enabled ? "已启用" : "未启用"],
                  ].map(([label, value]) => (
                    <div key={String(label)} className="rounded-2xl bg-canvas p-3">
                      <dt className="text-[11px] text-muted">{label}</dt>
                      <dd className="mt-1 break-all text-[13px] text-ink">{value}</dd>
                    </div>
                  ))}
                </dl>
                <p className="mt-3 text-[11px] leading-5 text-muted">
                  平台只读 HIS，不向其写入业务记录；需要院方开通的通道会在此标注「待开通」。
                </p>
              </section>

              <section className="card overflow-hidden">
                <div className="px-4 py-3"><h2 className="section-title">能力矩阵（谁负责什么）</h2></div>
                <ul className="divide-y divide-line border-t border-line">
                  {his.capabilities.map((item) => (
                    <li key={item.name} className="flex flex-wrap items-center gap-2 px-4 py-2 text-[12px]">
                      <span className="badge bg-canvas text-muted">{item.owner}</span>
                      <span className="font-medium text-ink">{item.name}</span>
                      <span className="text-muted">{item.note}</span>
                    </li>
                  ))}
                </ul>
              </section>

              <section className="card overflow-hidden">
                <div className="px-4 py-3"><h2 className="section-title">最近连通性检查</h2></div>
                {his.checks.length === 0 ? (
                  <div className="border-t border-line p-4">
                    <StateBlock kind="empty" title="还没有检查记录" description="点上面的「测试连接」会留痕" />
                  </div>
                ) : (
                  <ul className="divide-y divide-line border-t border-line">
                    {his.checks.map((item, index) => (
                      <li key={index} className="flex flex-wrap items-center gap-2 px-4 py-2 text-[12px]">
                        <span className={item.ok ? "badge bg-emerald-50 text-emerald-700" : "badge bg-rose-50 text-rose-700"}>
                          {item.ok ? "正常" : "异常"}
                        </span>
                        <span className="text-ink">{item.detail}</span>
                        <span className="text-muted">{item.latency_ms ?? "—"}ms</span>
                        <span className="ml-auto text-[11px] text-muted">{item.actor || "—"} · {formatTime(item.created_at)}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </section>
            </>
          ) : null}

          {tab === "tenant" ? (
            <>
              <section className="card p-4">
                <h2 className="section-title">新建院区</h2>
                <div className="mt-2 flex flex-wrap items-end gap-2">
                  <input
                    value={tenantCode}
                    onChange={(event) => setTenantCode(event.target.value)}
                    placeholder="编码（小写）"
                    className="rounded-xl bg-canvas px-3 py-2 text-[12px] text-ink"
                  />
                  <input
                    value={tenantName}
                    onChange={(event) => setTenantName(event.target.value)}
                    placeholder="院区名称"
                    className="rounded-xl bg-canvas px-3 py-2 text-[12px] text-ink"
                  />
                  <select
                    value={tenantIsolation}
                    onChange={(event) => setTenantIsolation(event.target.value)}
                    className="rounded-xl bg-canvas px-3 py-2 text-[12px] text-ink"
                  >
                    <option value="logical">逻辑隔离</option>
                    <option value="physical">物理隔离</option>
                  </select>
                  <button
                    onClick={() =>
                      void guard(async () => {
                        const created = await createTenant({
                          code: tenantCode.trim(),
                          name: tenantName.trim(),
                          isolation: tenantIsolation,
                        });
                        setTenantCode("");
                        setTenantName("");
                        setNotice("已创建院区 " + created.tenant.name);
                        await load();
                      })
                    }
                    disabled={busy || !tenantCode.trim() || !tenantName.trim()}
                    className="rounded-full bg-primary px-4 py-2 text-[12px] font-medium text-white disabled:opacity-50"
                  >
                    创建
                  </button>
                </div>
              </section>

              <section className="card overflow-hidden">
                <div className="px-4 py-3"><h2 className="section-title">院区列表</h2></div>
                <ul className="divide-y divide-line border-t border-line">
                  {tenants.map((tenant) => (
                    <li key={tenant.id} className="flex flex-wrap items-center gap-2 px-4 py-2 text-[12px]">
                      <span className="font-medium text-ink">{tenant.name}</span>
                      <span className="font-mono text-[11px] text-muted">{tenant.code}</span>
                      <span className="badge bg-canvas text-muted">
                        {tenant.isolation === "physical" ? "物理隔离" : "逻辑隔离"}
                      </span>
                      <span className={tenant.status === "active" ? "badge bg-emerald-50 text-emerald-700" : "badge bg-amber-50 text-amber-700"}>
                        {STATUS_LABEL[tenant.status] || tenant.status}
                      </span>
                      <span className="ml-auto flex gap-2">
                        <button
                          onClick={() =>
                            void guard(async () => {
                              await updateTenant(tenant.id, {
                                status: tenant.status === "active" ? "suspended" : "active",
                              });
                              await load();
                            })
                          }
                          disabled={busy}
                          className="text-[12px] font-medium text-primary disabled:opacity-50"
                        >
                          {tenant.status === "active" ? "暂停" : "恢复"}
                        </button>
                      </span>
                    </li>
                  ))}
                </ul>
                <p className="border-t border-line px-4 py-2 text-[11px] text-muted">
                  说明：平台做路由与展示，租户数据隔离的执行（INT-06）尚未接入，已在验收自测里标为阻塞。
                </p>
              </section>
            </>
          ) : null}

          {tab === "experiment" ? (
            <>
              <section className="card p-4">
                <h2 className="section-title">新建实验</h2>
                <div className="mt-2 flex flex-wrap items-end gap-2">
                  <input
                    value={expKey}
                    onChange={(event) => setExpKey(event.target.value)}
                    placeholder="key（小写）"
                    className="rounded-xl bg-canvas px-3 py-2 text-[12px] text-ink"
                  />
                  <input
                    value={expName}
                    onChange={(event) => setExpName(event.target.value)}
                    placeholder="实验名"
                    className="rounded-xl bg-canvas px-3 py-2 text-[12px] text-ink"
                  />
                  <input
                    value={expVariants}
                    onChange={(event) => setExpVariants(event.target.value)}
                    placeholder="实验组，逗号分隔"
                    className="w-48 rounded-xl bg-canvas px-3 py-2 text-[12px] text-ink"
                  />
                  <button
                    onClick={() =>
                      void guard(async () => {
                        await createExperiment({
                          key: expKey.trim(),
                          name: expName.trim(),
                          variants: expVariants.split(",").map((item) => item.trim()).filter(Boolean),
                        });
                        setExpKey("");
                        setExpName("");
                        setNotice("实验已创建（草稿态，灰度 0%）");
                        await load();
                      })
                    }
                    disabled={busy || !expKey.trim() || !expName.trim()}
                    className="rounded-full bg-primary px-4 py-2 text-[12px] font-medium text-white disabled:opacity-50"
                  >
                    创建
                  </button>
                </div>
              </section>

              <section className="card overflow-hidden">
                <div className="px-4 py-3"><h2 className="section-title">实验列表</h2></div>
                {experiments.length === 0 ? (
                  <div className="border-t border-line p-4">
                    <StateBlock kind="empty" title="还没有实验" description="先建一个草稿实验，再从 20% 灰度开始" />
                  </div>
                ) : (
                  <ul className="divide-y divide-line border-t border-line">
                    {experiments.map((item) => (
                      <li key={item.id} className="px-4 py-3 text-[12px]">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-medium text-ink">{item.name}</span>
                          <span className="font-mono text-[11px] text-muted">{item.key}</span>
                          <span className={item.status === "running" ? "badge bg-emerald-50 text-emerald-700" : "badge bg-canvas text-muted"}>
                            {STATUS_LABEL[item.status] || item.status}
                          </span>
                          <span className="text-muted">灰度 {item.rollout}%</span>
                          <span className="text-muted">组：{item.variants.join(" / ")}</span>
                          <span className="ml-auto flex items-center gap-2">
                            <input
                              type="number"
                              min={0}
                              max={100}
                              defaultValue={item.rollout}
                              aria-label={"灰度比例 " + item.key}
                              onChange={(event) => {
                                const value = Number(event.target.value);
                                void guard(async () => {
                                  await updateExperiment(item.key, {
                                    status: value > 0 ? "running" : item.status,
                                    rollout: value,
                                  });
                                  await load();
                                });
                              }}
                              className="w-20 rounded-xl bg-canvas px-2 py-1 text-[12px] text-ink"
                            />
                            <button
                              onClick={() =>
                                void guard(async () => {
                                  await updateExperiment(item.key, {
                                    status: item.status === "running" ? "paused" : "running",
                                  });
                                  await load();
                                })
                              }
                              disabled={busy || (item.status !== "running" && item.rollout === 0)}
                              className="text-[12px] font-medium text-primary disabled:opacity-50"
                            >
                              {item.status === "running" ? "暂停" : "启动"}
                            </button>
                          </span>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
                <p className="border-t border-line px-4 py-2 text-[11px] text-muted">
                  运行态灰度必须大于 0；实验分配记录（assignment）暂未接入业务链路，当前只做配置与统计。
                </p>
              </section>
            </>
          ) : null}
        </div>
      )}
    </main>
  );
}
