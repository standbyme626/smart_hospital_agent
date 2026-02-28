export const RUNTIME_SETTINGS_STORAGE_KEY = "smart_hospital.runtime_settings.v1";

export interface RuntimeSettings {
  debugIncludeNodes: string[];
  topK: number;
  rerankThreshold: number;
  rewriteTimeout: number;
  crisisFastlane: boolean;
}

export const DEFAULT_RUNTIME_SETTINGS: RuntimeSettings = {
  debugIncludeNodes: ["Query_Rewrite", "Hybrid_Retriever", "Decision_Judge"],
  topK: 3,
  rerankThreshold: 0.15,
  rewriteTimeout: 4,
  crisisFastlane: true,
};

function clampInt(value: unknown, min: number, max: number, fallback: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, Math.round(parsed)));
}

function clampFloat(value: unknown, min: number, max: number, fallback: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, Number(parsed.toFixed(3))));
}

function parseNodes(raw: unknown): string[] {
  if (Array.isArray(raw)) {
    return Array.from(
      new Set(
        raw
          .map((item) => String(item || "").trim())
          .filter(Boolean)
          .slice(0, 20),
      ),
    );
  }
  if (typeof raw === "string") {
    return Array.from(
      new Set(
        raw
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean)
          .slice(0, 20),
      ),
    );
  }
  return [];
}

export function normalizeRuntimeSettings(input: unknown): RuntimeSettings {
  const obj = input && typeof input === "object" ? (input as Record<string, unknown>) : {};
  const parsedNodes = parseNodes(obj.debugIncludeNodes);

  return {
    debugIncludeNodes: parsedNodes.length ? parsedNodes : [...DEFAULT_RUNTIME_SETTINGS.debugIncludeNodes],
    topK: clampInt(obj.topK, 1, 10, DEFAULT_RUNTIME_SETTINGS.topK),
    rerankThreshold: clampFloat(obj.rerankThreshold, 0, 1, DEFAULT_RUNTIME_SETTINGS.rerankThreshold),
    rewriteTimeout: clampFloat(obj.rewriteTimeout, 1, 10, DEFAULT_RUNTIME_SETTINGS.rewriteTimeout),
    crisisFastlane: typeof obj.crisisFastlane === "boolean" ? obj.crisisFastlane : DEFAULT_RUNTIME_SETTINGS.crisisFastlane,
  };
}

export function loadRuntimeSettings(): RuntimeSettings {
  if (typeof window === "undefined") return { ...DEFAULT_RUNTIME_SETTINGS };
  try {
    const raw = window.localStorage.getItem(RUNTIME_SETTINGS_STORAGE_KEY);
    if (!raw) return { ...DEFAULT_RUNTIME_SETTINGS };
    return normalizeRuntimeSettings(JSON.parse(raw));
  } catch {
    return { ...DEFAULT_RUNTIME_SETTINGS };
  }
}

export function saveRuntimeSettings(settings: RuntimeSettings): RuntimeSettings {
  const normalized = normalizeRuntimeSettings(settings);
  if (typeof window !== "undefined") {
    window.localStorage.setItem(RUNTIME_SETTINGS_STORAGE_KEY, JSON.stringify(normalized));
  }
  return normalized;
}

