"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const ITEMS = [
  { href: "/clinician/", label: "诊室工作台" },
  { href: "/clinician/print/", label: "文书打印" },
  { href: "/clinician/followup/", label: "诊毕随访" },
  { href: "/clinician/review/", label: "药师审方" },
  { href: "/clinician/schedule/", label: "我的排班" },
  { href: "/clinician/ward/", label: "移动查房" },
  { href: "/clinician/todo/", label: "待办" },
];

export function ClinicianNav() {
  const pathname = usePathname() || "";
  return (
    <nav className="flex flex-wrap gap-2">
      {ITEMS.map((item) => {
        const base = item.href.replace(/\/$/, "");
        const active = base === "/clinician" ? pathname === "/clinician" || pathname === "/clinician/" : pathname.startsWith(base);
        return (
          <Link
            key={item.href}
            href={item.href}
            className={
              active
                ? "rounded-full bg-primary px-3 py-1 text-[11px] font-medium text-white"
                : "rounded-full bg-white px-3 py-1 text-[11px] text-muted ring-1 ring-line hover:text-primary"
            }
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
