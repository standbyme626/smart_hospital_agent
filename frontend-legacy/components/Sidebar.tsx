"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Activity,
  CalendarClock,
  ClipboardList,
  LogOut,
  MessageSquarePlus,
  ShieldCheck,
  X,
} from "lucide-react";
import clsx from "clsx";
import type { Session } from "@/lib/types";
import { clearToken } from "@/lib/api";

type Props = {
  sessions: Session[];
  activeSession: string | null;
  open: boolean;
  onClose: () => void;
  onNew: () => void;
  onSelect: (id: string) => void;
};

const nav = [
  { href: "/", label: "智能问诊", icon: MessageSquarePlus },
  { href: "/booking", label: "挂号支付", icon: CalendarClock },
  { href: "/admin", label: "运营后台", icon: ClipboardList },
];

export function Sidebar({ sessions, activeSession, open, onClose, onNew, onSelect }: Props) {
  const pathname = usePathname();
  const router = useRouter();

  return (
    <>
      {open ? (
        <button
          aria-label="关闭侧栏"
          className="fixed inset-0 z-30 bg-black/50 backdrop-blur-sm lg:hidden"
          onClick={onClose}
        />
      ) : null}
      <aside
        className={clsx(
          "fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-line bg-panel/95 p-4 transition-transform lg:static lg:z-auto lg:translate-x-0",
          open ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2.5">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-primary2 text-slate-950">
              <Activity size={18} strokeWidth={2.5} />
            </span>
            <span>
              <span className="block text-sm font-semibold text-white">大医 · 智能分诊</span>
              <span className="block text-[11px] text-slate-400">Smart Hospital Agent v2</span>
            </span>
          </Link>
          <button className="btn-ghost h-9 w-9 p-0 lg:hidden" onClick={onClose} aria-label="关闭">
            <X size={16} />
          </button>
        </div>

        <nav className="mt-6 space-y-1">
          {nav.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={clsx(
                  "flex items-center gap-2.5 rounded-xl px-3 py-2.5 text-sm transition",
                  active
                    ? "bg-primary/10 text-primary"
                    : "text-slate-300 hover:bg-panel2 hover:text-white",
                )}
                onClick={onClose}
              >
                <item.icon size={17} />
                {item.label}
              </Link>
            );
          })}
        </nav>

        <button className="btn-primary mt-4 w-full" onClick={onNew}>
          <MessageSquarePlus size={16} />
          新建问诊
        </button>

        <div className="mt-6 flex items-center justify-between px-1 text-[11px] uppercase tracking-wider text-slate-500">
          <span>最近会话</span>
          <span>{sessions.length}</span>
        </div>
        <div className="scrollbar-thin mt-2 flex-1 space-y-1 overflow-y-auto">
          {sessions.length === 0 ? (
            <p className="px-2 py-6 text-xs text-slate-500">还没有会话，发一条消息开始。</p>
          ) : (
            sessions.map((session) => (
              <button
                key={session.id}
                onClick={() => {
                  onSelect(session.id);
                  onClose();
                }}
                className={clsx(
                  "block w-full truncate rounded-lg px-2.5 py-2 text-left text-xs transition",
                  activeSession === session.id
                    ? "bg-panel2 text-white"
                    : "text-slate-400 hover:bg-panel2 hover:text-slate-200",
                )}
                title={session.title || session.id}
              >
                {session.title || session.id}
              </button>
            ))
          )}
        </div>

        <div className="mt-3 flex items-center justify-between rounded-xl border border-line bg-panel2 px-3 py-2 text-[11px] text-slate-400">
          <span className="inline-flex items-center gap-1.5">
            <ShieldCheck size={13} className="text-mint" />
            RBAC / 审计 / HITL
          </span>
          <button
            className="inline-flex items-center gap-1 text-slate-400 hover:text-rose"
            onClick={() => {
              clearToken();
              router.push("/login");
            }}
          >
            <LogOut size={13} />
            退出
          </button>
        </div>
      </aside>
    </>
  );
}
