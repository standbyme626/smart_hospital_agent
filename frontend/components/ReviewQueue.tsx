"use client";

import { useCallback, useEffect, useState } from "react";
import { Check, Loader2, RefreshCw, X } from "lucide-react";
import clsx from "clsx";
import { approveReview, getReviews, rejectReview } from "@/lib/api";
import type { ReviewRequest } from "@/lib/types";

export function ReviewQueue({ onChanged }: { onChanged?: () => void }) {
  const [reviews, setReviews] = useState<ReviewRequest[]>([]);
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setReviews(await getReviews("pending"));
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function decide(review: ReviewRequest, action: "approve" | "reject") {
    setBusy(review.id);
    setError(null);
    try {
      if (action === "approve") {
        const ok = window.confirm(
          `确认批准这笔支付？\n成员：${review.member_name || review.user_id}\n科室：${review.department || "未知"}\n金额：¥${review.amount ?? 0}\n订单：${review.order_id || review.ref_id}`,
        );
        if (!ok) return;
        await approveReview(review.id);
      } else {
        await rejectReview(review.id, notes[review.id] || "");
      }
      await load();
      onChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "操作失败");
    } finally {
      setBusy(null);
    }
  }

  return (
    <section className="card overflow-hidden">
      <div className="flex items-center justify-between border-b border-line px-4 py-3">
        <div>
          <p className="text-[12px] font-medium text-muted">待审核支付申请</p>
          <p className="mt-0.5 text-[11px] text-muted">批准后由服务端调用医院 MCP 完成支付；拒绝后订单回到未支付。</p>
        </div>
        <button onClick={load} disabled={loading} className="btn-ghost">
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} />刷新
        </button>
      </div>

      {error ? <div className="border-b border-line bg-rose-50 px-4 py-2 text-[11px] text-rose-700">{error}</div> : null}

      {loading && reviews.length === 0 ? (
        <div className="flex items-center gap-2 px-4 py-6 text-[12px] text-muted">
          <Loader2 size={14} className="animate-spin" />加载中…
        </div>
      ) : reviews.length === 0 ? (
        <p className="px-4 py-6 text-[12px] text-muted">暂无待审核申请。</p>
      ) : (
        <div className="divide-y divide-line/60">
          {reviews.map((review) => (
            <div key={review.id} className="px-4 py-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="badge bg-amber-50 text-amber-700">待审核</span>
                <span className="text-[13px] font-semibold text-ink">{review.department || "未知科室"}</span>
                <span className="text-[12px] text-muted">¥{review.amount ?? 0}</span>
                <span className="text-[11px] text-muted">订单 {review.order_id || review.ref_id}</span>
                <span className="text-[11px] text-muted">患者 {review.username || review.user_id}</span>
                {review.member_name ? <span className="text-[11px] text-muted">成员 {review.member_name}</span> : null}
                <span className="text-[11px] text-muted">申请于 {new Date(review.created_at).toLocaleString()}</span>
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <input
                  className="input max-w-xs"
                  placeholder="拒批备注（可选）"
                  value={notes[review.id] || ""}
                  onChange={(event) => setNotes((prev) => ({ ...prev, [review.id]: event.target.value }))}
                />
                <button
                  onClick={() => decide(review, "approve")}
                  disabled={busy === review.id}
                  className={clsx("btn-primary", busy === review.id ? "opacity-60" : "")}
                >
                  <Check size={14} />批准
                </button>
                <button
                  onClick={() => decide(review, "reject")}
                  disabled={busy === review.id}
                  className="btn-ghost text-rose-600"
                >
                  <X size={14} />拒绝
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
