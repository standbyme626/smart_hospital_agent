"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertTriangle,
  ChevronRight,
  Contrast,
  Download,
  Loader2,
  LogOut,
  Pencil,
  Plus,
  Save,
  Settings2,
  ShieldCheck,
  Trash2,
  Type,
  UserRound,
  Users,
  X,
} from "lucide-react";
import clsx from "clsx";
import { AppShell } from "@/components/AppShell";
import { useApp } from "@/lib/app-context";
import { createMember, deleteAccount, deleteMember, exportMyData, updateMember } from "@/lib/api";
import { POLICY_VERSION, relationOptions } from "@/lib/format";
import type { Member } from "@/lib/types";

type FormState = {
  name: string;
  relation: string;
  gender: string;
  birth_date: string;
  allergies: string;
  conditions: string;
  id_card: string;
};

const emptyForm: FormState = {
  name: "",
  relation: "父母",
  gender: "",
  birth_date: "",
  allergies: "",
  conditions: "",
  id_card: "",
};

export default function ProfilePage() {
  const { user, members, refreshFamily, settings, setSettings, logout, showToast } = useApp();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState<FormState>(emptyForm);
  const [open, setOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deletePassword, setDeletePassword] = useState("");
  const [deleting, setDeleting] = useState(false);
  const [consentVersion, setConsentVersion] = useState<string | null>(null);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem("sha_consent");
      if (!raw) return;
      const parsed = JSON.parse(raw) as { version?: unknown };
      if (typeof parsed?.version === "string") setConsentVersion(parsed.version);
    } catch {
      // ignore malformed consent record
    }
  }, []);

  async function exportData() {
    setExporting(true);
    try {
      const blob = await exportMyData();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "smart-hospital-data.zip";
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      showToast("数据导出已开始");
    } catch (err) {
      showToast(err instanceof Error ? err.message : "导出失败，请稍后再试");
    } finally {
      setExporting(false);
    }
  }

  async function confirmDeleteAccount() {
    if (!deletePassword) {
      showToast("请输入登录密码");
      return;
    }
    if (!window.confirm("确定永久删除账号？所有健康档案、报告图片和会话记录将无法恢复。")) return;
    setDeleting(true);
    try {
      await deleteAccount(deletePassword);
      showToast("账号已删除");
      logout();
    } catch (err) {
      showToast(err instanceof Error ? err.message : "删除失败，请稍后再试");
      setDeleting(false);
    }
  }

  function startEdit(member: Member) {
    setEditingId(member.id);
    setForm({
      name: member.name,
      relation: member.relation === "self" ? "本人" : member.relation,
      gender: member.gender || "",
      birth_date: member.birth_date || "",
      allergies: member.allergies || "",
      conditions: member.conditions || "",
      id_card: "",
    });
    setOpen(true);
  }

  function startAdd() {
    setEditingId(null);
    setForm(emptyForm);
    setOpen(true);
  }

  async function save() {
    if (!form.name.trim()) {
      showToast("请填写姓名");
      return;
    }
    setSaving(true);
    try {
      const payload = {
        name: form.name.trim(),
        relation: form.relation === "本人" ? "self" : form.relation,
        gender: form.gender || null,
        birth_date: form.birth_date || null,
        allergies: form.allergies || null,
        conditions: form.conditions || null,
        id_card: form.id_card.trim() || undefined,
        is_self: editingId ? members.find((item) => item.id === editingId)?.is_self || false : false,
      };
      if (editingId) await updateMember(editingId, payload);
      else await createMember(payload);
      await refreshFamily();
      showToast("已保存");
      setOpen(false);
      setEditingId(null);
      setForm(emptyForm);
    } catch (err) {
      showToast(err instanceof Error ? err.message : "保存失败");
    } finally {
      setSaving(false);
    }
  }

  async function remove(member: Member) {
    if (member.is_self) return;
    if (!window.confirm(`删除 ${member.name} 的家庭档案？`)) return;
    try {
      await deleteMember(member.id);
      await refreshFamily();
      showToast("已删除");
    } catch (err) {
      showToast(err instanceof Error ? err.message : "删除失败");
    }
  }

  return (
    <AppShell title="我的" subtitle="家庭档案、隐私与显示设置">
      <div className="mx-auto max-w-3xl space-y-4">
        <section className="card p-5">
          <div className="flex items-center gap-3">
            <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-emerald-400 text-white">
              <UserRound size={22} />
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-[16px] font-semibold text-ink">{user?.user || "用户"}</p>
              <p className="text-[12px] text-muted">{user?.role === "admin" ? "管理员" : "患者账号"} · 健康档案与问诊记录已同步</p>
            </div>
            {user?.role === "admin" ? (
              <Link href="/admin/" className="chip text-primary">运营后台<ChevronRight size={13} /></Link>
            ) : null}
          </div>
        </section>

        <section className="card p-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2"><Users size={17} className="text-primary" /><h2 className="section-title">家庭成员</h2></div>
            <button onClick={startAdd} className="chip text-primary"><Plus size={13} />添加</button>
          </div>
          <div className="mt-3 space-y-2">
            {members.map((member) => (
              <div key={member.id} className="flex items-center gap-3 rounded-2xl bg-canvas p-3">
                <span className="flex h-9 w-9 items-center justify-center rounded-full bg-white text-primary"><Users size={15} /></span>
                <div className="min-w-0 flex-1">
                  <p className="text-[13px] font-medium text-ink">
                    {member.name}
                    {member.name !== (member.relation === "self" ? "本人" : member.relation) ? (
                      <span className="ml-1.5 text-[11px] text-muted">{member.relation === "self" ? "本人" : member.relation}</span>
                    ) : null}
                  </p>
                  <p className="mt-0.5 truncate text-[11px] text-muted">
                    {[member.gender, member.birth_date, member.allergies ? `过敏：${member.allergies}` : "", member.conditions ? `既往：${member.conditions}` : ""].filter(Boolean).join(" · ") || "暂无档案信息"}
                  </p>
                  <p className="mt-0.5 text-[11px] text-muted">就诊人ID：{member.hospital_patient_id || "—"}</p>
                </div>
                <button onClick={() => startEdit(member)} className="flex h-11 w-11 items-center justify-center rounded-full text-muted hover:bg-white hover:text-primary" aria-label="编辑"><Pencil size={15} /></button>
                {!member.is_self ? (
                  <button onClick={() => remove(member)} className="flex h-11 w-11 items-center justify-center rounded-full text-muted hover:bg-white hover:text-rose-500" aria-label="删除"><Trash2 size={15} /></button>
                ) : null}
              </div>
            ))}
          </div>
        </section>

        {open ? (
          <section className="card p-5 fade-up">
            <div className="flex items-center justify-between">
              <h2 className="section-title">{editingId ? "编辑家庭成员" : "添加家庭成员"}</h2>
              <button onClick={() => setOpen(false)} className="text-muted hover:text-ink" aria-label="关闭"><X size={17} /></button>
            </div>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <label className="block"><span className="label">姓名</span><input className="input" value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} placeholder="例如：张三" /></label>
              <label className="block"><span className="label">关系</span><select className="input" value={form.relation} onChange={(event) => setForm({ ...form, relation: event.target.value })}>{["本人", ...relationOptions.filter((item) => item !== "self")].map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
              <label className="block sm:col-span-2">
                <span className="label">身份证号（挂号实名制需要）</span>
                <input
                  className="input"
                  value={form.id_card}
                  onChange={(event) => setForm({ ...form, id_card: event.target.value })}
                  placeholder={
                    editingId && members.find((item) => item.id === editingId)?.has_id_card
                      ? `已保存（尾号 ${members.find((item) => item.id === editingId)?.id_card_last4 || "****"}），留空则不修改`
                      : "例如：310101198801019876"
                  }
                />
                <span className="mt-1 block text-[11px] text-muted">
                  仅用于医院实名建档与挂号；App 侧只展示尾号。
                </span>
              </label>
              <label className="block"><span className="label">性别</span><select className="input" value={form.gender} onChange={(event) => setForm({ ...form, gender: event.target.value })}><option value="">未填写</option><option value="男">男</option><option value="女">女</option><option value="其他">其他</option></select></label>
              <label className="block"><span className="label">出生日期</span><input className="input" type="date" value={form.birth_date} onChange={(event) => setForm({ ...form, birth_date: event.target.value })} /></label>
              <label className="block sm:col-span-2"><span className="label">过敏史</span><input className="input" value={form.allergies} onChange={(event) => setForm({ ...form, allergies: event.target.value })} placeholder="例如：青霉素过敏" /></label>
              <label className="block sm:col-span-2"><span className="label">既往史 / 慢病</span><input className="input" value={form.conditions} onChange={(event) => setForm({ ...form, conditions: event.target.value })} placeholder="例如：高血压 10 年" /></label>
            </div>
            <div className="mt-4 flex gap-2">
              <button onClick={() => setOpen(false)} className="btn-ghost flex-1">取消</button>
              <button onClick={save} disabled={saving} className="btn-primary flex-1"><Save size={15} />保存</button>
            </div>
          </section>
        ) : null}

        <section className="card p-5">
          <div className="flex items-center gap-2"><Settings2 size={17} className="text-primary" /><h2 className="section-title">显示与无障碍</h2></div>
          <div className="mt-3 space-y-3">
            <SettingRow icon={Type} label="字体大小" description="适合长辈或阅读困难时使用" value={settings.fontScale} options={[["md", "标准"], ["lg", "大"]]} onChange={(value) => setSettings({ fontScale: value as "md" | "lg" })} />
            <SettingRow icon={Contrast} label="对比度" description="更强的前景/背景对比" value={settings.contrast} options={[["normal", "标准"], ["high", "高对比"]]} onChange={(value) => setSettings({ contrast: value as "normal" | "high" })} />
            <SettingRow icon={Activity} label="动效" description="减少过渡与动画" value={settings.motion} options={[["normal", "标准"], ["reduced", "减少"]]} onChange={(value) => setSettings({ motion: value as "normal" | "reduced" })} />
          </div>
        </section>

        <section className="card p-5">
          <div className="flex items-center gap-2"><ShieldCheck size={17} className="text-primary" /><h2 className="section-title">隐私与数据</h2></div>
          <p className="mt-2 text-[12px] leading-6 text-muted">
            你可以随时导出自己的全部健康数据（JSON + 报告图片），或永久删除账号及关联记录。
            家庭成员档案需要获得本人（未成年人则为监护人）授权后使用。
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <Link href="/terms/" className="chip text-primary">用户协议<ChevronRight size={13} /></Link>
            <Link href="/privacy/" className="chip text-primary">隐私政策<ChevronRight size={13} /></Link>
          </div>
          <p className="mt-2 text-[11px] leading-5 text-muted">
            当前同意版本：{consentVersion || "未记录（历史账号）"} · 最新版本 {POLICY_VERSION}
          </p>
          <p className="mt-1 text-[11px] leading-5 text-muted">
            撤回说明：如需撤回同意，可在此删除账号或联系管理员；撤回后将停止使用在线问诊、报告解读等服务，已有数据按《隐私政策》处理。
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button onClick={exportData} disabled={exporting} className="chip text-primary disabled:opacity-40">
              {exporting ? <Loader2 size={13} className="animate-spin" /> : <Download size={13} />}
              导出我的数据
            </button>
            <button onClick={() => setDeleteOpen((prev) => !prev)} className="chip text-rose-600">
              <Trash2 size={13} />
              删除账号
            </button>
          </div>
          {deleteOpen ? (
            <div className="mt-3 space-y-2 rounded-2xl border border-rose-200 bg-rose-50 p-3">
              <p className="flex items-center gap-1.5 text-[12px] font-medium text-rose-700">
                <AlertTriangle size={14} />删除后无法恢复
              </p>
              <p className="text-[11px] leading-5 text-rose-700">
                将永久删除账号、家庭成员档案、报告图片、用药、随访、会话与健康记录，并匿名化审计日志。
              </p>
              <input
                className="input"
                type="password"
                value={deletePassword}
                onChange={(event) => setDeletePassword(event.target.value)}
                placeholder="输入登录密码确认"
                autoComplete="current-password"
              />
              <div className="flex gap-2">
                <button
                  onClick={() => { setDeleteOpen(false); setDeletePassword(""); }}
                  className="btn-ghost flex-1"
                >
                  取消
                </button>
                <button
                  onClick={confirmDeleteAccount}
                  disabled={deleting || !deletePassword}
                  className="inline-flex min-h-11 flex-1 items-center justify-center gap-2 rounded-full bg-rose-600 px-4 text-sm font-medium text-white transition hover:bg-rose-700 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {deleting ? <Loader2 size={15} className="animate-spin" /> : <Trash2 size={15} />}
                  {deleting ? "删除中…" : "永久删除"}
                </button>
              </div>
            </div>
          ) : null}
          <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-muted">
            <span className="chip">最小化采集</span>
            <span className="chip">可导出</span>
            <span className="chip">可删除</span>
          </div>
        </section>

        <button onClick={logout} className="btn-ghost w-full"><LogOut size={15} />退出登录</button>
        <p className="pb-4 text-center text-[11px] text-muted">大医 · Smart Hospital Agent v3 · 2026 H2</p>
      </div>
    </AppShell>
  );
}

function SettingRow({
  icon: Icon,
  label,
  description,
  value,
  options,
  onChange,
}: {
  icon: typeof Type;
  label: string;
  description: string;
  value: string;
  options: [string, string][];
  onChange: (value: string) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-2xl bg-canvas p-3">
      <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-white text-primary"><Icon size={16} /></span>
      <div className="min-w-0 flex-1">
        <p className="text-[13px] font-medium text-ink">{label}</p>
        <p className="text-[11px] text-muted">{description}</p>
      </div>
      <div className="flex rounded-full bg-white p-1">
        {options.map(([optionValue, optionLabel]) => (
          <button
            key={optionValue}
            onClick={() => onChange(optionValue)}
            aria-pressed={value === optionValue}
            className={clsx("min-h-11 rounded-full px-3 text-[11px]", value === optionValue ? "bg-primary text-white" : "text-muted")}
          >
            {optionLabel}
          </button>
        ))}
      </div>
    </div>
  );
}
