"use client";

import type { ReactNode } from "react";
import { AppFrame } from "./AppFrame";

export function AppShell({
  title,
  subtitle,
  actions,
  children,
}: {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <AppFrame end="patient" title={title} subtitle={subtitle} actions={actions}>
      {children}
    </AppFrame>
  );
}
