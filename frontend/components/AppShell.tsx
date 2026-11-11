"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";
import {
  Activity,
  Bell,
  CalendarCheck2,
  CalendarClock,
  ChevronRight,
  ClipboardList,
  FileText,
  HeartPulse,
  Home,
  LogOut,
  Pill,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Stethoscope,
  UserRound,
} from "lucide-react";
import clsx from "clsx";
import { useApp } from "@/lib/app-context";
import { FamilySwitcher } from "./FamilySwitcher";

const navItems = [
  { href: "/", label: "首页", icon: Home },
  { href: "/messages", label: "消息", icon: Bell },
  { href: "/followups", label: "随访", icon: CalendarCheck2 },
  { href: "/consult", label: "AI 诊室", icon: Stethoscope },
  { href: "/reports", label: "报告解读", icon: FileText },
  { href: "/meds", label: "药管家", icon: Pill },
  { href: "/health", label: "健康", icon: HeartPulse },
  { href: "/services", label: "挂号服务", icon: CalendarClock },
];

const mobileNav = [
  { href: "/", label: "首页", icon: Home },
  { href: "/messages", label: "消息", icon: Bell },
  { href: "/consult", label: "问诊", icon: Stethoscope },
  { href: "/services", label: "服务", icon: CalendarClock },
  { href: "/health", label: "健康", icon: HeartPulse },
  { href: "/profile", label: "我的", icon: UserRound },
];

export function AppShell({
  title,
  subtitle,
  children,
  actions,
}: {
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  actions?: React.ReactNode;
}) {
  const { user, authReady, logout } = useApp();
  const pathname = usePathname();
  const router = useRouter();

  useEffect(() => {
    if (authReady && !user) {
      router.replace("/login/");
    }
  }, [authReady, user, router]);

  if (!authReady || !user) {
    return (
      <div className="flex min-h-dvh items-center justify-center">
        <div className="flex items-center gap-2 text-sm text-muted">
          <span className="h-2 w-2 rounded-full bg-primary pulse-soft" />
          正在加载…
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-dvh">
      <aside className="fixed inset-y-0 left-0 z-40 hidden w-64 flex-col border-r border-line bg-card/90 px-4 py-5 backdrop-blur lg:flex">
        <Link href="/" className="flex items-center gap-2.5 px-2">
          <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-emerald-400 text-white shadow-sm">
            <Activity size={20} strokeWidth={2.4} />
          </span>
          <span>
            <span className="block text-[15px] font-semibold text-ink">大医</span>
            <span className="block text-[11px] text-muted">有证据的就医助手</span>
          </span>
        </Link>

        <nav className="mt-7 space-y-1">
          {navItems.map((item) => {
            const active = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
            return (
              <Link
                key={item.href}
                href={item.href}
                className={clsx(
                  "flex min-h-11 items-center gap-3 rounded-2xl px-3 text-sm transition",
                  active ? "bg-primary-soft font-medium text-primary" : "text-muted hover:bg-canvas hover:text-ink",
                )}
              >
                <item.icon size={18} />
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="mt-auto space-y-1">
          {user.role === "admin" ? (
            <>
              <Link href="/ops" className="flex min-h-11 items-center gap-3 rounded-2xl px-3 text-sm text-muted hover:bg-canvas hover:text-ink">
                <ClipboardList size={18} />
                运营台
              </Link>
              <Link href="/platform" className="flex min-h-11 items-center gap-3 rounded-2xl px-3 text-sm text-muted hover:bg-canvas hover:text-ink">
                <SlidersHorizontal size={18} />
                平台管理
              </Link>
            </>
          ) : null}
          <Link href="/profile" className="flex min-h-11 items-center gap-3 rounded-2xl px-3 text-sm text-muted hover:bg-canvas hover:text-ink">
            <Settings size={18} />
            设置与家庭
          </Link>
          <button
            onClick={logout}
            className="flex min-h-11 w-full items-center gap-3 rounded-2xl px-3 text-sm text-muted hover:bg-canvas hover:text-ink"
          >
            <LogOut size={18} />
            退出登录
          </button>
          <div className="mt-3 rounded-2xl bg-primary-soft px-3 py-2.5 text-[11px] leading-5 text-primary">
            <p className="flex items-center gap-1.5 font-medium">
              <ShieldCheck size={12} />
              隐私与安全
            </p>
            <p className="mt-1 text-primary/80">健康数据仅用于本次服务；可在「设置与家庭」中导出或删除。</p>
          </div>
        </div>
      </aside>

      <div className="lg:pl-64">
        <header className="sticky top-0 z-30 border-b border-line/70 bg-canvas/85 px-4 py-3 backdrop-blur lg:px-8">
          <div className="mx-auto flex max-w-6xl items-center gap-3">
            <Link href="/" className="flex items-center gap-2 lg:hidden">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-emerald-400 text-white">
                <Activity size={17} strokeWidth={2.4} />
              </span>
              <span className="text-sm font-semibold">大医</span>
            </Link>
            {title ? <h1 className="min-w-0 flex-1 truncate text-[15px] font-semibold text-ink lg:hidden">{title}</h1> : null}
            <div className="hidden min-w-0 flex-1 lg:block">
              {title ? <h1 className="truncate text-[17px] font-semibold text-ink">{title}</h1> : null}
              {subtitle ? <p className="truncate text-[12px] text-muted">{subtitle}</p> : null}
            </div>
            <div className="ml-auto flex items-center gap-2">
              {actions}
              <FamilySwitcher />
              <Link
                href="/profile"
                className="flex h-10 w-10 items-center justify-center rounded-full bg-primary-soft text-primary"
                aria-label="个人中心"
              >
                <UserRound size={17} />
              </Link>
            </div>
          </div>
        </header>

        <main className="mobile-safe mx-auto max-w-6xl px-4 py-5 lg:px-8 lg:py-8">{children}</main>
      </div>

      <nav className="fixed inset-x-0 bottom-0 z-50 border-t border-line bg-card/95 pb-[env(safe-area-inset-bottom)] backdrop-blur lg:hidden">
        <div className="grid grid-cols-6">
          {mobileNav.map((item) => {
            const active = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
            return (
              <Link
                key={item.href}
                href={item.href}
                className={clsx(
                  "flex min-h-14 flex-col items-center justify-center gap-1 text-[10.5px]",
                  active ? "text-primary" : "text-muted",
                )}
              >
                <item.icon size={20} />
                {item.label}
              </Link>
            );
          })}
        </div>
      </nav>
    </div>
  );
}

export function BackLink({ href = "/", label = "返回" }: { href?: string; label?: string }) {
  return (
    <Link href={href} className="inline-flex items-center gap-1 text-[13px] text-muted hover:text-ink">
      <ChevronRight size={14} className="rotate-180" />
      {label}
    </Link>
  );
}
