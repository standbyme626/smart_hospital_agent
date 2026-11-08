"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Activity, ArrowRight, Loader2, LockKeyhole, ShieldCheck, UserRound } from "lucide-react";
import { useApp } from "@/lib/app-context";

export default function LoginPage() {
  const router = useRouter();
  const { user, authReady, login } = useApp();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (authReady && user) router.replace("/");
  }, [authReady, user, router]);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      await login(username.trim(), password);
      router.replace("/");
    } catch (err) {
      const message = err instanceof Error ? err.message : "";
      if (message.includes("401") || message.includes("unauthorized") || message.includes("invalid credentials")) {
        setError("账号或密码不正确，请重试");
      } else if (message.includes("429") || message.includes("rate limit")) {
        setError("请求过于频繁，请稍后再试");
      } else {
        setError("网络连接失败，请重试");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid min-h-dvh lg:grid-cols-[1.05fr_0.95fr]">
      <div className="hidden flex-col justify-between bg-primary p-10 text-white lg:flex">
        <div className="flex items-center gap-2.5">
          <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white/15">
            <Activity size={22} strokeWidth={2.4} />
          </span>
          <span>
            <span className="block text-[16px] font-semibold">大医</span>
            <span className="block text-[12px] text-white/70">有证据的就医助手</span>
          </span>
        </div>
        <div>
          <h1 className="max-w-md text-4xl font-semibold leading-tight">
            先分诊，<br />再决定下一步。
          </h1>
          <p className="mt-4 max-w-md text-[14px] leading-7 text-white/80">
            从症状到科室，从报告到用药，从挂号到就诊指引。每条建议都可以查看医学依据，危机情况直接走急诊通道。
          </p>
          <div className="mt-8 grid gap-3 text-[13px] text-white/85">
            <p className="flex items-center gap-2"><ShieldCheck size={15} /> 健康数据加密存储，可随时导出或删除</p>
            <p className="flex items-center gap-2"><ShieldCheck size={15} /> AI 不确诊、不开处方，危机情况转人工/急诊</p>
            <p className="flex items-center gap-2"><ShieldCheck size={15} /> 挂号支付走医院 MCP，支付前需人工确认</p>
          </div>
        </div>
        <p className="text-[11px] text-white/50">Smart Hospital Agent v3 · 2026 H2</p>
      </div>

      <div className="flex items-center justify-center px-5 py-10">
        <div className="w-full max-w-sm">
          <div className="mb-7 flex items-center gap-2.5 lg:hidden">
            <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-primary text-white">
              <Activity size={21} strokeWidth={2.4} />
            </span>
            <span>
              <span className="block text-[16px] font-semibold">大医</span>
              <span className="block text-[12px] text-muted">有证据的就医助手</span>
            </span>
          </div>
          <h2 className="text-2xl font-semibold text-ink">欢迎回来</h2>
          <p className="mt-1 text-[13px] text-muted">登录后继续你的健康档案与问诊记录。</p>

          <form className="mt-7 space-y-4" onSubmit={submit}>
            <label className="block">
              <span className="label">账号</span>
              <div className="relative">
                <UserRound size={16} className="absolute left-3.5 top-3.5 text-muted" />
                <input
                  className="input pl-10"
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  placeholder="请输入账号"
                  autoComplete="username"
                  required
                />
              </div>
            </label>
            <label className="block">
              <span className="label">密码</span>
              <div className="relative">
                <LockKeyhole size={16} className="absolute left-3.5 top-3.5 text-muted" />
                <input
                  className="input pl-10"
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="请输入密码"
                  autoComplete="current-password"
                  required
                />
              </div>
            </label>
            {error ? <p className="rounded-xl bg-rose-50 px-3 py-2 text-[12px] text-rose-600">{error}</p> : null}
            <button className="btn-primary w-full" disabled={loading} type="submit">
              {loading ? <Loader2 size={16} className="animate-spin" /> : <ArrowRight size={16} />}
              登录
            </button>
          </form>

          <p className="mt-5 text-center text-[11px] leading-5 text-muted">
            登录即表示同意《用户协议》与《隐私政策》。<br />
            演示账号请向管理员获取；管理员可使用后台账号登录。
          </p>
        </div>
      </div>
    </div>
  );
}
