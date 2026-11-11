import type { Metadata, Viewport } from "next";
import "./globals.css";
import { AppProvider } from "@/lib/app-context";
import { PwaRegister } from "@/components/PwaRegister";

export const metadata: Metadata = {
  title: "大医 · 有证据的就医助手",
  description: "分诊、报告解读、用药检查、挂号支付与健康档案，一个有证据的医疗 AI 助手。",
  manifest: "/manifest.webmanifest",
  applicationName: "大医",
  appleWebApp: { capable: true, title: "大医", statusBarStyle: "default" },
};

export const viewport: Viewport = {
  themeColor: "#0E9384",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>
        <AppProvider>
          {children}
          <PwaRegister />
        </AppProvider>
      </body>
    </html>
  );
}
