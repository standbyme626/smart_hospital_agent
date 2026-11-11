"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const ITEMS = [
  { href: "/ops/", label: "概览" },
  { href: "/ops/reviews/", label: "支付审核" },
  { href: "/ops/audit/", label: "审计复盘" },
  { href: "/ops/analytics/", label: "运营分析" },
  { href: "/ops/insurance/", label: "医保对账" },
  { href: "/ops/crisis/", label: "危机事件" },
  { href: "/ops/notifications/", label: "通知记录" },
];

export function OpsNav() {
  const pathname = usePathname() || "";
  return (
    <nav className="flex flex-wrap gap-2">
      {ITEMS.map((item) => {
        const active = item.href === "/ops/" ? pathname === "/ops" || pathname === "/ops/" : pathname.startsWith(item.href.replace(/\/$/, ""));
        return (
          <Link
            key={item.href}
            href={item.href}
            className={active ? "rounded-full bg-primary px-4 py-1.5 text-[12px] font-medium text-white" : "rounded-full bg-white px-4 py-1.5 text-[12px] font-medium text-ink ring-1 ring-line hover:text-primary"}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
