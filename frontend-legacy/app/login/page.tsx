"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Activity, ArrowLeft, Loader2, LogIn } from "lucide-react";
import { login } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("admin123456");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      await login(username, password);
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "登录失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-dvh items-center justify-center px-4">
      <div className="w-full max-w-md">
        <Link href="/" className="mb-6 inline-flex items-center gap-2 text-xs text-slate-400 hover:text-white">
          <ArrowLeft size={14} />
          返回问诊
        </Link>
        <div className="surface p-6">
          <div className="flex items-center gap-2.5">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-primary2 text-slate-950">
              <Activity size={19} strokeWidth={2.5} />
            </span>
            <div>
              <h1 className="text-base font-semibold text-white">登录大医</h1>
              <p className="text-xs text-slate-400">Smart Hospital Agent v2</p>
            </div>
          </div>

          <form className="mt-6 space-y-4" onSubmit={submit}>
            <label className="block text-xs text-slate-400">
              用户名
              <input
                className="input mt-1.5"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                autoComplete="username"
              />
            </label>
            <label className="block text-xs text-slate-400">
              密码
              <input
                className="input mt-1.5"
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                autoComplete="current-password"
              />
            </label>
            {error ? <p className="text-xs text-rose">{error}</p> : null}
            <button className="btn-primary w-full" disabled={loading} type="submit">
              {loading ? <Loader2 size={16} className="animate-spin" /> : <LogIn size={16} />}
              登录
            </button>
          </form>

          <div className="mt-5 rounded-xl border border-line bg-panel2 px-3 py-2.5 text-[11px] leading-5 text-slate-400">
            <p>演示账号：admin / admin123456（管理员）</p>
            <p>患者账号：guest / guest</p>
          </div>
        </div>
      </div>
    </div>
  );
}
