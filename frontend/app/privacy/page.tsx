import Link from "next/link";
import { Activity, ChevronLeft, ShieldCheck } from "lucide-react";
import { POLICY_EFFECTIVE_DATE, POLICY_VERSION } from "@/lib/format";

export default function PrivacyPage() {
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
            <ShieldCheck size={20} />
            <h1 className="text-2xl font-semibold text-ink">隐私政策</h1>
          </div>
          <p className="mt-2 text-[12px] leading-6 text-muted">
            版本：{POLICY_VERSION} · 生效日期：{POLICY_EFFECTIVE_DATE} · 适用产品：大医 Smart Hospital Agent
          </p>
          <div className="mt-5 rounded-2xl border border-amber-200 bg-amber-50 p-3 text-[12px] leading-6 text-amber-800">
            当前为演示 / 试点版本，文本用于说明真实的数据处理方式；正式上线前将由法务出具最终版本，届时以法务版本为准。
          </div>

          <div className="mt-6 space-y-6 text-[14px] leading-7 text-ink">
            <section>
              <h2 className="text-[16px] font-semibold">1. 我们收集哪些信息</h2>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-[13px] leading-6 text-muted">
                <li>账号信息：用户名、登录凭据（加密存储）、角色。</li>
                <li>健康档案：家庭成员姓名、关系、性别、出生日期、过敏史、既往史与就诊人编号。</li>
                <li>业务数据：问诊会话与结果、上传的报告 / 皮肤照片、用药清单、随访与挂号订单。</li>
                <li>使用记录：页面访问与操作事件、必要的错误日志（用于安全与稳定性）。</li>
              </ul>
            </section>

            <section>
              <h2 className="text-[16px] font-semibold">2. 数据用途</h2>
              <p className="mt-2 text-[13px] leading-6 text-muted">
                用于提供分诊、报告解读、用药相互作用检查、家庭健康档案、随访提醒与挂号服务；用于保障账号与系统安全；
                用于改进产品与服务质量。我们不会将你的健康数据用于广告投放，也不会出售你的个人信息。
              </p>
            </section>

            <section>
              <h2 className="text-[16px] font-semibold">3. 保存期限</h2>
              <p className="mt-2 text-[13px] leading-6 text-muted">
                账号与健康档案在账号存续期间保存；报告图片、问诊记录、用药与随访记录随账号保存，删除账号后同步删除。
                安全审计日志会保留必要期限后匿名化或删除。你可以在「我的 → 隐私与数据」随时导出或永久删除账号及关联数据。
              </p>
            </section>

            <section>
              <h2 className="text-[16px] font-semibold">4. 第三方模型与存储说明</h2>
              <p className="mt-2 text-[13px] leading-6 text-muted">
                报告解读、问诊分诊等能力会调用第三方 AI 模型服务（文本 / 视觉）。调用时仅发送完成本次处理所必需的内容，
                模型服务提供方按其服务条款处理数据。图片与结构化记录存储在本服务的数据库与文件存储中，不对外公开分享。
                具体模型提供方可能随版本调整，重大变更会在本页更新版本号。
              </p>
            </section>

            <section>
              <h2 className="text-[16px] font-semibold">5. 家庭成员与授权</h2>
              <p className="mt-2 text-[13px] leading-6 text-muted">
                你可以为家人建立档案。请确保你已获得被建档人（未成年人则为监护人）的授权，并仅在授权范围内使用其健康信息。
                如需撤回授权，可删除对应成员档案或联系管理员。
              </p>
            </section>

            <section>
              <h2 className="text-[16px] font-semibold">6. 你的权利</h2>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-[13px] leading-6 text-muted">
                <li>查询与导出：在「我的 → 隐私与数据」导出 JSON 数据与报告图片。</li>
                <li>删除：删除单条报告、成员档案，或永久删除整个账号及关联记录。</li>
                <li>撤回同意：停止使用服务或联系管理员；撤回后我们将停止相应处理，法律要求保留的记录除外。</li>
                <li>更正：在家庭成员编辑中更正姓名、出生日期、过敏史与既往史。</li>
              </ul>
            </section>

            <section>
              <h2 className="text-[16px] font-semibold">7. 数据安全</h2>
              <p className="mt-2 text-[13px] leading-6 text-muted">
                我们采用访问控制、加密传输、最小权限与审计日志等措施保护数据。互联网环境无法保证绝对安全，
                如发生安全事件，我们会按适用法律与内部流程处置并告知受影响用户。
              </p>
            </section>

            <section>
              <h2 className="text-[16px] font-semibold">8. 未成年人</h2>
              <p className="mt-2 text-[13px] leading-6 text-muted">
                未成年人档案应由监护人建立与管理。若你是未成年人，请在监护人陪同下使用本服务。
              </p>
            </section>

            <section>
              <h2 className="text-[16px] font-semibold">9. 联系方式</h2>
              <p className="mt-2 text-[13px] leading-6 text-muted">
                隐私相关咨询、导出或删除请求：请联系系统管理员（演示环境请通过部署方提供的联系方式）。
              </p>
            </section>
          </div>

          <p className="mt-6 border-t border-line pt-4 text-[11px] leading-5 text-muted">
            本页为 {POLICY_VERSION} 版本，自 {POLICY_EFFECTIVE_DATE} 起生效。当前为演示 / 试点，最终以法务版本为准。
          </p>
          <div className="mt-3 flex flex-wrap gap-2 text-[12px]">
            <Link href="/terms/" className="chip text-primary">查看《用户协议》</Link>
            <Link href="/login/" className="chip">返回登录</Link>
          </div>
        </div>
      </main>
    </div>
  );
}
