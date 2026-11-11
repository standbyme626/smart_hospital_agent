"use client";

import type { ReactNode } from "react";
import { AppFrame } from "./AppFrame";

export function ConsoleShell({
  end,
  title,
  subtitle,
  actions,
  children,
}: {
  end: "clinician" | "ops" | "platform";
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <AppFrame end={end} title={title} subtitle={subtitle} actions={actions}>
      {children}
    </AppFrame>
  );
}
