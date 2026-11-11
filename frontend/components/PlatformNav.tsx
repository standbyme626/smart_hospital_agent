"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const ITEMS = [
  { href: "/platform/", label: "运行总览" },
  { href: "/platform/config/", label: "配置管理" },
  { href: "/platform/knowledge/", label: "知识库" },
  { href: "/platform/insight/", label: "可观测" },
  { href: "/platform/integrations/", label: "接入与实验" },
  { href: "/platform/spec/", label: "规范与数据流" },
];

export function PlatformNav() {
  const pathname = usePathname() || "";
  return (
    <nav className="flex flex-wrap gap-2">
      {ITEMS.map((item) => {
        const active = item.href === "/platform/" ? pathname === "/platform" || pathname === "/platform/" : pathname.startsWith(item.href.replace(/\/$/, ""));
        return (
          <Link
            key={item.href}
            href={item.href}
            className={
              active
                ? "rounded-full bg-primary px-4 py-1.5 text-[12px] font-medium text-white"
                : "rounded-full bg-white px-4 py-1.5 text-[12px] font-medium text-ink ring-1 ring-line"
            }
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
