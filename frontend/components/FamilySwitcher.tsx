"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Check, ChevronDown, Plus, Users } from "lucide-react";
import clsx from "clsx";
import { useApp } from "@/lib/app-context";

export function FamilySwitcher() {
  const { members, activeMember, setActiveMemberId } = useApp();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    function onDoc(event: MouseEvent) {
      if (ref.current && !ref.current.contains(event.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((value) => !value)}
        className="flex min-h-10 items-center gap-2 rounded-full border border-line bg-card px-3 text-[13px] text-ink hover:bg-canvas"
      >
        <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary-soft text-primary">
          <Users size={13} />
        </span>
        <span className="max-w-[90px] truncate">{activeMember?.name || "本人"}</span>
        <ChevronDown size={14} className={clsx("text-muted transition", open ? "rotate-180" : "")} />
      </button>

      {open ? (
        <div className="absolute right-0 top-12 z-50 w-60 rounded-2xl border border-line bg-card p-2 shadow-xl">
          <p className="px-2 py-1 text-[11px] text-muted">为谁咨询</p>
          {members.map((member) => (
            <button
              key={member.id}
              onClick={() => {
                setActiveMemberId(member.id);
                setOpen(false);
              }}
              className="flex min-h-10 w-full items-center justify-between rounded-xl px-2 text-left text-[13px] text-ink hover:bg-canvas"
            >
              <span>
                {member.name}
                {member.name !== (member.relation === "self" ? "本人" : member.relation) ? (
                  <span className="ml-1.5 text-[11px] text-muted">{member.relation === "self" ? "本人" : member.relation}</span>
                ) : null}
              </span>
              {activeMember?.id === member.id ? <Check size={14} className="text-primary" /> : null}
            </button>
          ))}
          <Link
            href="/profile/"
            onClick={() => setOpen(false)}
            className="mt-1 flex min-h-10 items-center gap-2 rounded-xl px-2 text-[13px] text-primary hover:bg-primary-soft"
          >
            <Plus size={14} />
            管理家庭成员
          </Link>
        </div>
      ) : null}
    </div>
  );
}
