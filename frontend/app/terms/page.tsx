import Link from "next/link";
import { Activity, ChevronLeft, FileText } from "lucide-react";
import { POLICY_EFFECTIVE_DATE, POLICY_VERSION } from "@/lib/format";

export default function TermsPage() {
  return (
    <div className="min-h-dvh">
      <header className="border-b border-line bg-card/80 px-5 py-4 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-3">
          <Link href="/" className="flex items-center gap-2">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-emerald-400 text-white">
              <Activity size={17} strokeWidth={2.4} />
            </span>
            <span className="text-sm font-semibold">大医</span>
          </Link>
          <Link href="/login/" className="inline-flex items-center gap-1 text-[12px] text-muted hover:text-ink">
            <ChevronLeft size={14} />返回登录
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-5 py-8">
        <div className="card p-5 sm:p-7">
          <div className="flex items-center gap-2 text-primary">
            <FileText size={20} />
            <h1 className="text-2xl font-semibold text-ink">用户协议</h1>
          </div>
          <p className="mt-2 text-[12px] leading-6 text-muted">
            版本：{POLICY_VERSION} · 生效日期：{POLICY_EFFECTIVE_DATE} · 适用产品：大医 Smart Hospital Agent
          </p>
          <div className="mt-5 rounded-2xl border border-amber-200 bg-amber-50 p-3 text-[12px] leading-6 text-amber-800">
            当前为演示 / 试点版本，文本用于说明服务能力与边界；正式上线前将由法务出具最终版本，届时以法务版本为准。
          </div>

          <div className="mt-6 space-y-6 text-[14px] leading-7 text-ink">
            <section>
              <h2 className="text-[16px] font-semibold">1. 服务说明</h2>
              <p className="mt-2 text-[13px] leading-6 text-muted">
                大医提供 AI 分诊、报告与皮肤照片解读、用药清单与相互作用检查、家庭健康档案、随访与挂号辅助服务。
                服务可能随版本迭代调整，功能以实际提供为准。
              </p>
            </section>

            <section>
              <h2 className="text-[16px] font-semibold">2. 不构成医疗诊断</h2>
              <p className="mt-2 text-[13px] leading-6 text-muted">
                本服务提供的是健康信息整理与就医方向建议，不构成医生诊断、处方或治疗方案，不能替代线下面诊。
                任何用药调整、停药或检查决策，请先咨询具备资质的医生或药师。
              </p>
            </section>

            <section>
              <h2 className="text-[16px] font-semibold">3. 急诊与危险信号</h2>
              <p className="mt-2 text-[13px] leading-6 text-muted">
                出现胸痛伴出汗、呼吸困难、意识不清、大出血、抽搐等危险信号时，请立即拨打 120 或前往最近的急诊科，
                不要等待在线问诊结果。应用内的急诊提示与电话链接仅用于辅助，不替代急救系统。
              </p>
            </section>

            <section>
              <h2 className="text-[16px] font-semibold">4. 账号与安全</h2>
              <p className="mt-2 text-[13px] leading-6 text-muted">
                请妥善保管账号与密码，不要与他人共用。你对自己账号下发生的操作负责；发现异常登录请及时修改密码或联系管理员。
              </p>
            </section>

            <section>
              <h2 className="text-[16px] font-semibold">5. 使用规范</h2>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-[13px] leading-6 text-muted">
                <li>不得上传与健康服务无关的违法、侵权或恶意内容。</li>
                <li>不得冒用他人身份建档，或未经授权上传他人健康信息。</li>
                <li>不得攻击、探测或干扰本服务的正常运行。</li>
              </ul>
            </section>

            <section>
              <h2 className="text-[16px] font-semibold">6. 第三方 AI 与内容</h2>
              <p className="mt-2 text-[13px] leading-6 text-muted">
                部分能力由第三方 AI 模型提供，输出可能存在不准确或不完整之处。我们会结合本地知识库与安全规则进行约束，
                但不对模型输出的绝对准确性作出保证。医学依据仅在知识库检索命中时展示。
              </p>
            </section>

            <section>
              <h2 className="text-[16px] font-semibold">7. 挂号与支付</h2>
              <p className="mt-2 text-[13px] leading-6 text-muted">
                挂号支付需经管理员人工审核后生效，批准前不会扣款。演示 / 试点环境的号源、医生与支付信息可能为模拟数据，
                不构成真实医疗机构承诺。
              </p>
            </section>

            <section>
              <h2 className="text-[16px] font-semibold">8. 责任限制</h2>
              <p className="mt-2 text-[13px] leading-6 text-muted">
                因网络、设备、第三方服务故障或不可抗力导致的服务中断，我们将尽力恢复但不承担超出法律规定的责任。
                因依赖本服务建议而延误就医造成的后果，本服务不承担责任。
              </p>
            </section>

            <section>
              <h2 className="text-[16px] font-semibold">9. 协议变更与联系方式</h2>
              <p className="mt-2 text-[13px] leading-6 text-muted">
                协议更新会在本页发布新版本并更新版本号与生效日期。继续使用即视为接受更新后的协议。
                如有疑问，请联系系统管理员（演示环境请通过部署方提供的联系方式）。
              </p>
            </section>
          </div>

          <p className="mt-6 border-t border-line pt-4 text-[11px] leading-5 text-muted">
            本页为 {POLICY_VERSION} 版本，自 {POLICY_EFFECTIVE_DATE} 起生效。当前为演示 / 试点，最终以法务版本为准。
          </p>
          <div className="mt-3 flex flex-wrap gap-2 text-[12px]">
            <Link href="/privacy/" className="chip text-primary">查看《隐私政策》</Link>
            <Link href="/login/" className="chip">返回登录</Link>
          </div>
        </div>
      </main>
    </div>
  );
}
