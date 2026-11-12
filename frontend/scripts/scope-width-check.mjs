import { chromium } from "@playwright/test";
import fs from "node:fs";
const BASE = "https://hospital.kk.zhzepei.cn";
const env = Object.fromEntries(fs.readFileSync("/home/kkk/Project/smart_hospital_agent_v2/.env", "utf-8").split("\n").map((l) => l.match(/^([A-Z0-9_]+)=(.+)$/)).filter(Boolean).map((m) => [m[1], m[2].trim()]));
const browser = await chromium.launch();
async function login(user, pass) {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, locale: "zh-CN" });
  const page = await ctx.newPage();
  await page.goto(BASE + "/login/", { waitUntil: "networkidle" });
  await page.getByPlaceholder("请输入账号").fill(user);
  await page.getByPlaceholder("请输入密码").fill(pass);
  const c = page.getByRole("checkbox", { name: /我已阅读并同意/ });
  if (await c.count()) await c.check();
  await page.getByRole("button", { name: "登录", exact: true }).click();
  await page.waitForTimeout(2500);
  return { ctx, page };
}
const admin = await login(env.ADMIN_USER, env.ADMIN_PASSWORD);
for (const route of ["/", "/followups/", "/reports/", "/messages/", "/platform/"]) {
  await admin.page.goto(BASE + route, { waitUntil: "networkidle" });
  const m = await admin.page.evaluate(() => {
    const inner = document.querySelector(".app-main > div");
    const readable = document.querySelector(".readable");
    const panel = document.querySelector(".agent-panel-head")?.innerText.split("\n")[0] || null;
    const summary = document.querySelector(".agent-panel-title")?.parentElement?.innerText.split("\n").slice(0, 2).join(" / ");
    return {
      content: inner ? Math.round(inner.getBoundingClientRect().width) : null,
      readable: readable ? Math.round(readable.getBoundingClientRect().width) : null,
      panel,
      summary,
    };
  });
  console.log(route, "内容宽:", m.readable ?? m.content, "| 面板:", m.panel);
}
const patient = await login(env.PATIENT_USER, env.PATIENT_PASSWORD);
const pbody = await patient.page.evaluate(async () => {
  const token = window.localStorage.getItem("sha_token");
  const r = await fetch("/api/v1/agent/panel", { headers: { Authorization: "Bearer " + token } });
  return (await r.json()).scope;
});
console.log("患者端面板 scope:", pbody);
await browser.close();
