import { chromium } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
const BASE = "https://hospital.kk.zhzepei.cn";
const OUT = "/tmp/shell-shots2";
fs.mkdirSync(OUT, { recursive: true });
const env = Object.fromEntries(fs.readFileSync("/home/kkk/Project/smart_hospital_agent_v2/.env", "utf-8").split("\n").map((l) => l.match(/^([A-Z0-9_]+)=(.+)$/)).filter(Boolean).map((m) => [m[1], m[2].trim()]));
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, locale: "zh-CN" });
const page = await ctx.newPage();
await page.goto(BASE + "/login/", { waitUntil: "networkidle" });
await page.getByPlaceholder("请输入账号").fill(env.ADMIN_USER);
await page.getByPlaceholder("请输入密码").fill(env.ADMIN_PASSWORD);
const c = page.getByRole("checkbox", { name: /我已阅读并同意/ });
if (await c.count()) await c.check();
await page.getByRole("button", { name: "登录", exact: true }).click();
await page.getByRole("heading", { name: "哪里不舒服？" }).waitFor({ timeout: 30000 });
const routes = ["/", "/clinician/", "/ops/", "/platform/"];
for (const route of routes) {
  await page.goto(BASE + route, { waitUntil: "networkidle" });
  const info = await page.evaluate(() => ({
    panel: document.querySelector(".agent-panel-head")?.innerText.split("\n")[0] || null,
    sider: document.querySelectorAll(".sider-link").length,
    h1: document.querySelector("h1")?.innerText || null,
  }));
  console.log(route, "| 面板:", info.panel, "| 导航项:", info.sider, "| H1:", info.h1);
  await page.screenshot({ path: path.join(OUT, "d" + route.replace(/\//g, "_") + ".png") });
}
await browser.close();
