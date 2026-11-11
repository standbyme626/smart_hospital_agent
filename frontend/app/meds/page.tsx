"use client";

import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Loader2,
  Pill,
  Plus,
  ShieldAlert,
  Trash2,
} from "lucide-react";
import clsx from "clsx";
import { AppShell } from "@/components/AppShell";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { useApp } from "@/lib/app-context";
import {
  checkMedications,
  createMedication,
  deleteMedication,
  getMedications,
  toggleMedication,
  track,
} from "@/lib/api";
import { severityMeta } from "@/lib/format";
import type { Medication } from "@/lib/types";

type CheckResult = {
  drugs: string[];
  interactions: { drug_a: string; drug_b: string; severity: string; description: string }[];
  pregnancy_contraindications: { drug: string; description: string }[];
  highest_severity: string;
  coverage?: string;
  unrecognized?: string[];
  included_medications?: string[];
};

export default function MedsPage() {
  const { activeMember, showToast } = useApp();
  const [meds, setMeds] = useState<Medication[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);
  const [name, setName] = useState("");
  const [dose, setDose] = useState("");
  const [frequency, setFrequency] = useState("");
  const [saving, setSaving] = useState(false);
  const [checkText, setCheckText] = useState("");
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<CheckResult | null>(null);

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeMember?.id]);

  async function load() {
    setLoading(true);
    setLoadError(false);
    try {
      setMeds(await getMedications(activeMember?.id));
    } catch {
      setMeds([]);
      setLoadError(true);
    } finally {
      setLoading(false);
    }
  }

  async function add() {
    if (!name.trim()) {
      showToast("请填写药品名称");
      return;
    }
    setSaving(true);
    try {
      await createMedication({ name: name.trim(), dose, frequency, member_id: activeMember?.id });
      setName("");
      setDose("");
      setFrequency("");
      showToast("已加入用药清单");
      track("medication_add", { name });
      await load();
    } catch {
      showToast("保存失败，请稍后再试");
    } finally {
      setSaving(false);
    }
  }

  async function toggle(item: Medication) {
    try {
      await toggleMedication(item.id, !item.active);
      await load();
    } catch {
      showToast("操作失败");
    }
  }

  async function remove(item: Medication) {
    if (!window.confirm(`删除 ${item.name}？`)) return;
    try {
      await deleteMedication(item.id);
      await load();
    } catch {
      showToast("删除失败");
    }
  }

  async function check() {
    if (!checkText.trim()) {
      showToast("请描述要检查的药物或症状");
      return;
    }
    setChecking(true);
    setResult(null);
    try {
      const data = await checkMedications(checkText, activeMember?.id);
      setResult(data);
      track("medication_check", { highest: data.highest_severity });
    } catch {
      showToast("检查失败，请稍后再试");
    } finally {
      setChecking(false);
    }
  }

  const activeMeds = meds.filter((item) => item.active);
  const inactiveMeds = meds.filter((item) => !item.active);

  return (
    <AppShell title="药管家" subtitle="用药清单、相互作用与妊娠禁忌检查" actions={<SourceBadge live />}>
      <div className="mx-auto max-w-4xl space-y-4">
        <OfflineBanner />
        <section className="card p-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-primary"><Pill size={18} /><h2 className="text-[17px] font-semibold">正在服用</h2></div>
            <span className="text-[11px] text-muted">{loading || loadError ? "—" : `${activeMeds.length} 种`}</span>
          </div>
          <div className="mt-4 space-y-2">
            {loading ? <div className="h-16 animate-pulse rounded-2xl bg-canvas" /> : null}
            {!loading && loadError ? <StateBlock kind="error" onRetry={load} /> : null}
            {!loading && !loadError && activeMeds.length === 0 ? (
              <StateBlock kind="empty" title="还没有用药记录" description="添加药品后，可以自动带上清单做相互作用检查" />
            ) : null}
            {activeMeds.map((item) => (
              <div key={item.id} className="flex items-center gap-3 rounded-2xl bg-canvas p-3">
                <span className="flex h-9 w-9 items-center justify-center rounded-full bg-emerald-50 text-emerald-600"><Pill size={16} /></span>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[14px] font-medium text-ink">{item.name}</p>
                  <p className="mt-0.5 text-[11px] text-muted">{item.dose || "剂量未填写"} · {item.frequency || "频次未填写"}</p>
                </div>
                <button onClick={() => toggle(item)} className="text-[11px] text-muted hover:text-primary">停用</button>
                <button onClick={() => remove(item)} className="text-muted hover:text-rose-500" aria-label="删除"><Trash2 size={15} /></button>
              </div>
            ))}
          </div>

          {inactiveMeds.length ? (
            <div className="mt-4">
              <p className="text-[11px] text-muted">已停用</p>
              <div className="mt-2 flex flex-wrap gap-2">
                {inactiveMeds.map((item) => (
                  <button key={item.id} onClick={() => toggle(item)} className="chip text-muted">
                    {item.name} · 恢复
                  </button>
                ))}
              </div>
            </div>
          ) : null}
        </section>

        <section className="card p-5">
          <h2 className="section-title">添加用药</h2>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <label className="block"><span className="label">药品名称</span><input className="input" value={name} onChange={(event) => setName(event.target.value)} placeholder="例如：华法林" /></label>
            <label className="block"><span className="label">剂量</span><input className="input" value={dose} onChange={(event) => setDose(event.target.value)} placeholder="例如：3mg" /></label>
            <label className="block sm:col-span-2"><span className="label">频次</span><input className="input" value={frequency} onChange={(event) => setFrequency(event.target.value)} placeholder="例如：每日一次，晚餐后" /></label>
          </div>
          <button onClick={add} disabled={saving} className="btn-primary mt-4">
            {saving ? <Loader2 size={15} className="animate-spin" /> : <Plus size={15} />}
            加入清单
          </button>
        </section>

        <section className="card p-5">
          <div className="flex items-center gap-2 text-primary"><ShieldAlert size={18} /><h2 className="text-[17px] font-semibold">相互作用检查</h2></div>
          <p className="mt-1 text-[12px] leading-5 text-muted">
            会自动带上当前用药清单。也可以补充新药、保健品或想问的组合。
          </p>
          {activeMeds.length ? (
            <div className="mt-3 flex flex-wrap gap-2">
              {activeMeds.map((item) => <span key={item.id} className="chip">{item.name}</span>)}
            </div>
          ) : null}
          <textarea
            className="input mt-3 min-h-[96px]"
            value={checkText}
            onChange={(event) => setCheckText(event.target.value)}
            placeholder="例如：牙痛想吃布洛芬；或者：阿司匹林 + 华法林能一起用吗？"
          />
          <button onClick={check} disabled={checking} className="btn-primary mt-3">
            {checking ? <Loader2 size={15} className="animate-spin" /> : <ShieldAlert size={15} />}
            检查
          </button>

          {result ? (() => {
            const severity = severityMeta(result.highest_severity);
            const partial = String(result.coverage || "").toLowerCase() === "partial" || Boolean(result.unrecognized?.length);
            const noneButPartial = partial && severity.key === "none";
            return (
              <div className="mt-4 space-y-2">
                <div className={clsx("rounded-2xl border p-3 text-[13px]", noneButPartial ? "border-amber-200 bg-amber-50 text-amber-800" : severity.className)}>
                  <p className="flex items-center gap-1.5 font-medium">
                    {noneButPartial || severity.dangerous ? <AlertTriangle size={14} /> : severity.key === "none" ? <CheckCircle2 size={14} /> : <ShieldAlert size={14} />}
                    {noneButPartial ? "仅检查了部分药物，不能据此判断安全" : `最高风险：${severity.label}`}
                  </p>
                  <p className="mt-1 text-[12px] opacity-80">
                    识别到的药物：{result.drugs.length ? result.drugs.join("、") : "未识别"}
                  </p>
                  {result.included_medications?.length ? (
                    <p className="mt-1 text-[12px] opacity-80">已纳入检查：{result.included_medications.join("、")}</p>
                  ) : null}
                  {result.unrecognized?.length ? (
                    <p className="mt-1 text-[12px] opacity-80">未识别：{result.unrecognized.join("、")}（请核对药名或咨询药师）</p>
                  ) : null}
                  {partial ? (
                    <p className="mt-1 text-[12px] font-medium opacity-90">仅检查了部分药物或存在未识别药物，不能据此判断安全。</p>
                  ) : null}
                </div>
                {result.interactions.map((item, index) => {
                  const itemSeverity = severityMeta(item.severity);
                  return (
                    <div key={index} className="rounded-2xl border border-line bg-canvas p-3">
                      <p className="flex flex-wrap items-center gap-2 text-[13px] font-medium text-ink">
                        <span>{item.drug_a} + {item.drug_b}</span>
                        <span className={clsx("badge border", itemSeverity.className)}>
                          {itemSeverity.dangerous ? <AlertTriangle size={11} /> : itemSeverity.key === "none" ? <CheckCircle2 size={11} /> : <ShieldAlert size={11} />}
                          {itemSeverity.label}
                        </span>
                      </p>
                      <p className="mt-1 text-[12px] leading-5 text-muted">{item.description}</p>
                    </div>
                  );
                })}
                {result.pregnancy_contraindications.map((item, index) => (
                  <div key={index} className="rounded-2xl border border-rose-200 bg-rose-50 p-3 text-[12px] leading-5 text-rose-700">
                    <p className="flex items-center gap-1.5 font-medium"><AlertTriangle size={12} />妊娠禁忌：{item.drug}</p>
                    <p className="mt-1">{item.description}</p>
                  </div>
                ))}
              </div>
            );
          })() : null}
        </section>

        <p className="text-center text-[11px] leading-5 text-muted">
          相互作用结果来自本地规则表，不能覆盖所有药物组合；具体用药请咨询医生或药师。
        </p>
      </div>
    </AppShell>
  );
}
