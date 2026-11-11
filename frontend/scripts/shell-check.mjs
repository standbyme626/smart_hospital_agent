import { chromium } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const BASE = "https://hospital.kk.zhzepei.cn";
const OUT = "/tmp/shell-shots";
fs.mkdirSync(OUT, { recursive: true });
const env = Object.fromEntries(
  fs.readFileSync("/home/kkk/Project/smart_hospital_agent_v2/.env", "utf-8")
    .split("\n").map((l) => l.match(/^([A-Z0-9_]+)=(.+)$/)).filter(Boolean)
    .map((m) => [m[1], m[2].trim()]),
);

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, locale: "zh-CN" });
const page = await ctx.newPage();
const bad = [];
page.on("response", (r) => { if (r.status() >= 400) bad.push(r.status() + " " + r.url()); });
await page.goto(BASE + "/login/", { waitUntil: "networkidle" });
await page.getByPlaceholder("请输入账号").fill(env.ADMIN_USER);
await page.getByPlaceholder("请输入密码").fill(env.ADMIN_PASSWORD);
const consent = page.getByRole("checkbox", { name: /我已阅读并同意/ });
if (await consent.count()) await consent.check();
await page.getByRole("button", { name: "登录", exact: true }).click();
await page.getByRole("heading", { name: "哪里不舒服？" }).waitFor({ timeout: 30000 });

const paths = ["/", "/messages/", "/clinician/", "/ops/", "/platform/", "/platform/config/"];
for (const route of paths) {
  await page.goto(BASE + route, { waitUntil: "networkidle" });
  const m = await page.evaluate(() => {
    const r = (sel) => { const el = document.querySelector(sel); if (!el) return null; const b = el.getBoundingClientRect(); return Math.round(b.x) + "/" + Math.round(b.width); };
    return { main: r(".app-main"), panel: r(".agent-panel"), sider: r(".app-sider"), h1: document.querySelector("h1")?.innerText || null, overflow: document.documentElement.scrollWidth - window.innerWidth };
  });
  console.log(route, "sider:" + m.sider, "main:" + m.main, "panel:" + m.panel, "h1:" + m.h1, "溢出:" + m.overflow);
  await page.screenshot({ path: path.join(OUT, "d" + route.replace(/\//g, "_") + ".png") });
}

const mob = await browser.newContext({ viewport: { width: 390, height: 844 }, locale: "zh-CN" });
const mp = await mob.newPage();
const token = await page.evaluate(() => window.localStorage.getItem("sha_token"));
await mp.addInitScript((tk) => window.localStorage.setItem("sha_token", tk), token);
for (const route of ["/", "/messages/"]) {
  await mp.goto(BASE + route, { waitUntil: "networkidle" });
  const m = await mp.evaluate(() => ({
    overflow: document.documentElement.scrollWidth - window.innerWidth,
    tabs: document.querySelectorAll(".mobile-tabbar a").length,
    siderVisible: !!document.querySelector(".app-sider") && getComputedStyle(document.querySelector(".app-sider")).position,
  }));
  console.log("mobile", route, "溢出:" + m.overflow, "底栏:" + m.tabs, "sider:" + m.siderVisible);
  await mp.screenshot({ path: path.join(OUT, "m" + route.replace(/\//g, "_") + ".png") });
}
console.log("4xx/5xx:", bad.length, bad.slice(0, 4));
await browser.close();
