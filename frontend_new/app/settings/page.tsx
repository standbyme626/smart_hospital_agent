"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import {
  DEFAULT_RUNTIME_SETTINGS,
  type RuntimeSettings,
  loadRuntimeSettings,
  normalizeRuntimeSettings,
  saveRuntimeSettings,
} from "@/lib/runtime-settings";

function toNodeText(nodes: string[]): string {
  return nodes.join(", ");
}

export default function SettingsPage() {
  const initial = useMemo(() => loadRuntimeSettings(), []);
  const [settings, setSettings] = useState<RuntimeSettings>(initial);
  const [nodesText, setNodesText] = useState<string>(toNodeText(initial.debugIncludeNodes));
  const [savedAt, setSavedAt] = useState<string>("");

  const onSave = () => {
    const normalized = normalizeRuntimeSettings({
      ...settings,
      debugIncludeNodes: nodesText,
    });
    const persisted = saveRuntimeSettings(normalized);
    setSettings(persisted);
    setNodesText(toNodeText(persisted.debugIncludeNodes));
    setSavedAt(new Date().toLocaleTimeString());
  };

  const onReset = () => {
    const persisted = saveRuntimeSettings(DEFAULT_RUNTIME_SETTINGS);
    setSettings(persisted);
    setNodesText(toNodeText(persisted.debugIncludeNodes));
    setSavedAt(new Date().toLocaleTimeString());
  };

  return (
    <main className="settings-page">
      <section className="settings-hero">
        <div>
          <h1>运行参数设置</h1>
          <p>这些参数会注入 `/api/v1/chat/stream` 请求，并在 SSE/Langfuse 显示最终生效值。</p>
        </div>
        <Link className="settings-back" href="/">
          返回会话
        </Link>
      </section>

      <section className="settings-panel">
        <label className="settings-field">
          <span>debug_include_nodes（逗号分隔）</span>
          <input value={nodesText} onChange={(e) => setNodesText(e.target.value)} placeholder="Query_Rewrite, Hybrid_Retriever" />
        </label>

        <div className="settings-grid">
          <label className="settings-field">
            <span>top_k</span>
            <input
              type="number"
              min={1}
              max={10}
              value={settings.topK}
              onChange={(e) => setSettings((prev) => ({ ...prev, topK: Number(e.target.value || 3) }))}
            />
          </label>

          <label className="settings-field">
            <span>rerank_threshold</span>
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={settings.rerankThreshold}
              onChange={(e) => setSettings((prev) => ({ ...prev, rerankThreshold: Number(e.target.value || 0.15) }))}
            />
          </label>

          <label className="settings-field">
            <span>rewrite_timeout（秒）</span>
            <input
              type="number"
              min={1}
              max={10}
              step={0.5}
              value={settings.rewriteTimeout}
              onChange={(e) => setSettings((prev) => ({ ...prev, rewriteTimeout: Number(e.target.value || 4) }))}
            />
          </label>
        </div>

        <label className="settings-check">
          <input
            type="checkbox"
            checked={settings.crisisFastlane}
            onChange={(e) => setSettings((prev) => ({ ...prev, crisisFastlane: e.target.checked }))}
          />
          <span>crisis_fastlane（危机意图启用快速专线）</span>
        </label>

        <div className="settings-actions">
          <button type="button" onClick={onSave}>
            保存并生效
          </button>
          <button type="button" className="secondary" onClick={onReset}>
            恢复默认
          </button>
          <p>{savedAt ? `已保存: ${savedAt}` : "尚未保存"}</p>
        </div>
      </section>
    </main>
  );
}

