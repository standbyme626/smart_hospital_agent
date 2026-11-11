"use client";

import { useEffect, useState, type ReactNode } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Activity, CalendarClock, HeartPulse, Home, LogOut, Menu, MessageSquare, Stethoscope, UserRound, X } from "lucide-react";
import { useApp } from "@/lib/app-context";
import { END_LABEL, NAV, isActive, type EndKey } from "@/lib/nav";
import { AgentPanel } from "./AgentPanel";
import { FamilySwitcher } from "./FamilySwitcher";

const PATIENT_TABS = [
  { href: "/", label: "首页", icon: Home },
  { href: "/messages/", label: "消息", icon: MessageSquare },
  { href: "/consult/", label: "问诊", icon: Stethoscope },
  { href: "/services/", label: "服务", icon: CalendarClock },
  { href: "/health/", label: "健康", icon: HeartPulse },
  { href: "/profile/", label: "我的", icon: UserRound },
];

type Props = {
  end: EndKey;
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  hideAgent?: boolean;
};

/** 演示站同款三栏页框：顶栏 64 + 左栏 208 + 主区 + 右栏 328 Agent 面板。 */
export function AppFrame({ end, title, subtitle, actions, children, hideAgent = false }: Props) {
  const pathname = usePathname() || "";
  const router = useRouter();
  const { user, logout } = useApp();
  const [siderOpen, setSiderOpen] = useState(false);
  const [agentOpen, setAgentOpen] = useState(false);

  useEffect(() => {
    setSiderOpen(false);
    setAgentOpen(false);
  }, [pathname]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setSiderOpen(false);
        setAgentOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const groups = NAV[end];

  return (
    <div className="app-shell">
      <a href="#main" className="skip-link">跳到主内容</a>

      <header className="app-topbar">
        <button
          type="button"
          className="icon-btn lg:hidden"
          aria-label="打开导航"
          aria-expanded={siderOpen}
          onClick={() => setSiderOpen((v) => !v)}
        >
          {siderOpen ? <X size={18} /> : <Menu size={18} />}
        </button>

        <Link href="/" className="brand">
          <span className="brand-logo"><Activity size={16} strokeWidth={2.6} /></span>
          <span className="brand-text">
            大医
            <small>有证据的就医助手</small>
          </span>
        </Link>

        <span className="scope-chip">{END_LABEL[end]}</span>

        <div className="topbar-right">
          {!hideAgent ? (
            <button
              type="button"
              className="icon-btn xl:hidden"
              aria-label="打开 Agent 面板"
              aria-expanded={agentOpen}
              onClick={() => setAgentOpen((v) => !v)}
            >
              <MessageSquare size={18} />
            </button>
          ) : null}
          {end === "patient" ? (
            <span className="family-slot hidden sm:inline-flex">
              <FamilySwitcher />
            </span>
          ) : null}
          <span className="role-chip">{user?.role === "admin" ? "scope: admin" : "scope: " + (user?.role || "patient")}</span>
          <Link href="/profile" className="avatar" aria-label="个人中心">
            <UserRound size={16} />
          </Link>
          <button
            type="button"
            className="icon-btn"
            aria-label="退出登录"
            onClick={() => {
              logout();
              router.push("/login/");
            }}
          >
            <LogOut size={17} />
          </button>
        </div>
      </header>

      <div className="app-grid">
        <aside className={"app-sider" + (siderOpen ? " open" : "")} aria-label="主导航">
          {groups.map((group) => (
            <nav key={group.title} aria-label={group.title}>
              <p className="sider-title">{group.title}</p>
              {group.items.map((item) => {
                const active = isActive(pathname, item.href);
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="sider-link"
                    aria-current={active ? "page" : undefined}
                  >
                    <span className="sider-dot" aria-hidden="true" />
                    {item.label}
                  </Link>
                );
              })}
            </nav>
          ))}
          <p className="sider-note">
            {end === "patient"
              ? "健康数据仅用于本次服务；可在「设置与家庭」导出或删除。"
              : "所有临床/配置写入都会留痕，危险操作需二次验证。"}
          </p>
        </aside>

        <main id="main" className="app-main">
          {title ? (
            <div className="page-head">
              <div className="min-w-0">
                <h1>{title}</h1>
                {subtitle ? <p>{subtitle}</p> : null}
              </div>
              {actions ? <div className="page-head-actions">{actions}</div> : null}
            </div>
          ) : null}
          {children}
        </main>

        {hideAgent ? null : <AgentPanel end={end} className={agentOpen ? "open" : ""} />}
      </div>

      {end === "patient" ? (
        <nav className="mobile-tabbar" aria-label="底部导航">
          {PATIENT_TABS.map((tab) => {
            const active = isActive(pathname, tab.href);
            return (
              <Link key={tab.href} href={tab.href} aria-current={active ? "page" : undefined}>
                <tab.icon size={19} />
                <span>{tab.label}</span>
              </Link>
            );
          })}
        </nav>
      ) : null}

      {(siderOpen || agentOpen) && !hideAgent ? (
        <button
          type="button"
          className="sheet-backdrop xl:hidden"
          aria-label="关闭面板"
          onClick={() => {
            setSiderOpen(false);
            setAgentOpen(false);
          }}
        />
      ) : null}
    </div>
  );
}

export function PageHead({
  title,
  subtitle,
  actions,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="page-head">
      <div className="min-w-0">
        <h1>{title}</h1>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
      {actions ? <div className="page-head-actions">{actions}</div> : null}
    </div>
  );
}
