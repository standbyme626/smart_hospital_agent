"use client";

import Link from "next/link";
import { Activity, ArrowRight, ShieldCheck, SlidersHorizontal, Stethoscope } from "lucide-react";

const ENTRIES = [
  {
    href: "/",
    icon: Activity,
    title: "患者端",
    desc: "分诊、报告解读、用药检查、挂号缴费、随访与消息",
    note: "患者账号登录",
    tone: "bg-primary-soft text-primary",
  },
  {
    href: "/clinician/",
    icon: Stethoscope,
    title: "医生工作台",
    desc: "今日队列、患者 360、病历草稿（签名默认关闭）",
    note: "doctor / admin 角色",
    tone: "bg-sky-50 text-sky-700",
  },
  {
    href: "/ops/",
    icon: ShieldCheck,
    title: "运营 / 质控台",
    desc: "支付异常、对账差异、人工接管",
    note: "ops / admin 角色",
    tone: "bg-amber-50 text-amber-700",
  },
  {
    href: "/platform/",
    icon: SlidersHorizontal,
    title: "平台管理",
    desc: "运行总览、配置管理（规则/模板/Prompt）、知识库与检索、可观测、接入与实验、规范与数据流",
    note: "admin 角色",
    tone: "bg-violet-50 text-violet-700",
  },
];

export default function EntriesPage() {
  return (
    <main className="min-h-dvh bg-canvas px-4 py-10">
      <div className="mx-auto max-w-4xl">
        <header className="text-center">
          <p className="text-[12px] uppercase tracking-[0.2em] text-muted">Smart Hospital Agent</p>
          <h1 className="mt-2 text-[26px] font-semibold text-ink">选择一个入口</h1>
          <p className="mt-2 text-[13px] text-muted">四端独立鉴权、独立布局；未登录会跳转登录页，角色不足会显示无权限。</p>
        </header>

        <div className="mt-8 grid gap-4 sm:grid-cols-2">
          {ENTRIES.map((item) => {
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className="card group flex items-start gap-3 p-5 transition hover:-translate-y-0.5 hover:shadow-md"
              >
                <span className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl ${item.tone}`}>
                  <Icon size={19} />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-1.5 text-[15px] font-semibold text-ink">
                    {item.title}
                    <ArrowRight size={14} className="text-muted transition group-hover:translate-x-0.5" />
                  </span>
                  <span className="mt-1 block text-[12px] leading-5 text-muted">{item.desc}</span>
                  <span className="mt-2 inline-flex rounded-full bg-canvas px-2 py-0.5 text-[11px] text-muted">{item.note}</span>
                </span>
              </Link>
            );
          })}
        </div>

        <div className="mt-8 flex flex-wrap items-center justify-center gap-3 text-[12px] text-muted">
          <Link href="/preview/" className="rounded-full bg-white px-4 py-2 ring-1 ring-line hover:text-primary">演示站（61 屏原型，模拟数据）</Link>
          <Link href="/login/" className="rounded-full bg-white px-4 py-2 ring-1 ring-line hover:text-primary">登录</Link>
          <Link href="/privacy/" className="hover:text-primary">隐私政策</Link>
          <Link href="/terms/" className="hover:text-primary">用户协议</Link>
        </div>

        <p className="mt-6 text-center text-[11px] leading-5 text-muted">
          医生签名与真实 HIS 写入默认关闭（需 HIS 账号透传）；规则 / 模板 / Prompt 已支持在线草稿→发布→回滚，
          灰度实验可配比例但分流执行未接；医保、短信 / 微信等外部通道待院方开通。
        </p>
      </div>
    </main>
  );
}
