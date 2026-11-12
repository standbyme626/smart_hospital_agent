"use client";

import { useEffect, useMemo, useState } from "react";
import { ShieldCheck } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { useApp } from "@/lib/app-context";
import { getConsentScopes, saveConsentScope, type ConsentScopeItem } from "@/lib/api";

export default function ConsentPage() {
  const { members, activeMember } = useApp();
  const [scopes, setScopes] = useState<ConsentScopeItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setFailed(false);
    try {
      setScopes(await getConsentScopes());
    } catch {
      setScopes([]);
      setFailed(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  const scopeMap = useMemo(() => new Map(scopes.map((item) => [item.member_id, item])), [scopes]);

  async function toggle(memberId: string, field: "can_view" | "can_act") {
    const current = scopeMap.get(memberId);
    const next = {
      member_id: memberId,
      can_view: current?.can_view ?? true,
      can_act: current?.can_act ?? false,
    };
    next[field] = !next[field];
    setBusy(memberId + ":" + field);
    try {
      const saved = await saveConsentScope(next);
      setScopes((prev) => {
        const rest = prev.filter((item) => item.member_id !== memberId);
        return [saved, ...rest];
      });
    } finally {
      setBusy(null);
    }
  }

  return (
    <AppShell title="家庭授权" subtitle="谁可以查看资料、谁可以代办" actions={<SourceBadge live />}>
      <div className="readable space-y-4">
        <OfflineBanner />
        <p className="rounded-2xl bg-canvas px-3 py-2 text-[12px] leading-5 text-muted">
          查看：能否读取成员的就诊、报告与用药；代办：能否代挂号、代缴费。关闭即撤回，立即生效。
        </p>
        {loading ? <div className="card h-24 animate-pulse" /> : failed ? (
          <div className="card p-3"><StateBlock kind="error" onRetry={load} /></div>
        ) : members.length === 0 ? (
          <div className="card p-3"><StateBlock kind="empty" title="还没有家庭成员" description="先在「我的」添加家人，再设置授权" /></div>
        ) : (
          <section className="card divide-y divide-line">
            {members.map((member) => {
              const scope = scopeMap.get(member.id);
              const canView = scope?.can_view ?? true;
              const canAct = scope?.can_act ?? false;
              const isSelf = member.is_self || member.id === activeMember?.id;
              return (
                <div key={member.id} className="flex flex-wrap items-center gap-3 px-4 py-3">
                  <span className="flex h-9 w-9 items-center justify-center rounded-full bg-primary-soft text-[12px] font-medium text-primary">
                    {member.name.slice(0, 1)}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="flex items-center gap-2 text-[13px] font-medium text-ink">
                      {member.name}
                      {isSelf ? <span className="badge">本人</span> : null}
                    </p>
                    <p className="mt-0.5 text-[11px] text-muted">{member.relation || "家人"} · 授权版本 {scope?.version || "未设置"}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => toggle(member.id, "can_view")}
                      disabled={busy === member.id + ":can_view"}
                      className={canView ? "chip border-primary bg-primary text-white" : "chip text-muted"}
                    >
                      查看 {canView ? "开" : "关"}
                    </button>
                    <button
                      onClick={() => toggle(member.id, "can_act")}
                      disabled={busy === member.id + ":can_act"}
                      className={canAct ? "chip border-primary bg-primary text-white" : "chip text-muted"}
                    >
                      代办 {canAct ? "开" : "关"}
                    </button>
                  </div>
                </div>
              );
            })}
          </section>
        )}
        <p className="flex items-center gap-1.5 text-center text-[11px] leading-5 text-muted">
          <ShieldCheck size={13} /> 授权变更留痕；涉及未成年人需监护人确认。
        </p>
      </div>
    </AppShell>
  );
}
