"use client";

import Link from "next/link";
import type { LucideIcon } from "lucide-react";
import clsx from "clsx";

export function TaskCard({
  href,
  onClick,
  icon: Icon,
  title,
  description,
  tone = "primary",
}: {
  href?: string;
  onClick?: () => void;
  icon: LucideIcon;
  title: string;
  description: string;
  tone?: "primary" | "amber" | "rose" | "sky" | "violet" | "emerald";
}) {
  const tones = {
    primary: "bg-primary-soft text-primary",
    amber: "bg-amber-50 text-amber-600",
    rose: "bg-rose-50 text-rose-500",
    sky: "bg-sky-50 text-sky-600",
    violet: "bg-violet-50 text-violet-600",
    emerald: "bg-emerald-50 text-emerald-600",
  };
  const content = (
    <div className="card flex h-full min-h-[118px] flex-col gap-3 p-4 text-left transition hover:-translate-y-0.5 hover:border-primary/30">
      <span className={clsx("flex h-10 w-10 items-center justify-center rounded-2xl", tones[tone])}>
        <Icon size={19} />
      </span>
      <span>
        <span className="block text-[14px] font-semibold text-ink">{title}</span>
        <span className="mt-1 block text-[12px] leading-5 text-muted">{description}</span>
      </span>
    </div>
  );
  if (href) return <Link href={href}>{content}</Link>;
  return (
    <button type="button" onClick={onClick} className="h-full w-full">
      {content}
    </button>
  );
}
