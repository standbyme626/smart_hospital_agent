import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "大医 · 智能分诊",
  description: "Medical triage agent with evidence-backed retrieval, drug interaction checks and MCP booking/payment.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
