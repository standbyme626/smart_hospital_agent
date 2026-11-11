import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");

function read(relative) {
  return fs.readFileSync(path.join(ROOT, relative), "utf-8");
}

test("PWA manifest 字段齐全（可安装的最低要求）", () => {
  const manifest = JSON.parse(read("public/manifest.webmanifest"));
  assert.equal(manifest.name, "大医 · 有证据的就医助手");
  assert.equal(manifest.short_name, "大医");
  assert.equal(manifest.start_url, "/");
  assert.equal(manifest.scope, "/");
  assert.equal(manifest.display, "standalone");
  assert.equal(manifest.lang, "zh-CN");
  assert.match(manifest.theme_color, /^#[0-9A-Fa-f]{6}$/);
  assert.match(manifest.background_color, /^#[0-9A-Fa-f]{6}$/);
  assert.ok(Array.isArray(manifest.icons) && manifest.icons.length >= 1, "至少一个图标");
  for (const icon of manifest.icons) {
    assert.ok(icon.src.startsWith("/"), "图标必须用站内绝对路径");
    assert.ok(icon.sizes && icon.type, "图标要有 sizes 与 type");
    assert.ok(fs.existsSync(path.join(ROOT, "public", icon.src)), "图标文件必须存在：" + icon.src);
  }
});

test("service worker 存在且不缓存业务数据", () => {
  const sw = read("public/sw.js");
  assert.match(sw, /addEventListener\("install"/);
  assert.match(sw, /addEventListener\("activate"/);
  assert.match(sw, /addEventListener\("fetch"/);
  assert.ok(!/caches\.open/.test(sw), "当前明确不透传缓存，避免脏数据");
});

test("layout 注册 manifest 与 service worker", () => {
  const layout = read("app/layout.tsx");
  assert.match(layout, /manifest:\s*"\/manifest\.webmanifest"/);
  assert.match(layout, /<PwaRegister\s*\/>/);

  const register = read("components/PwaRegister.tsx");
  assert.match(register, /serviceWorker\.register\(["']\/sw\.js["']\)/);
});

test("移动端导航可达（PWA 主要入口）", () => {
  const shell = read("components/AppShell.tsx");
  for (const keyword of ["首页", "消息", "随访", "AI 诊室"]) {
    assert.ok(shell.includes(keyword), "移动导航缺少入口：" + keyword);
  }
});
