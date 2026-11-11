import { chromium } from "@playwright/test";
import fs from "node:fs";
const BASE = "https://hospital.kk.zhzepei.cn";
const env = Object.fromEntries(fs.readFileSync("/home/kkk/Project/smart_hospital_agent_v2/.env", "utf-8").split("\n").map((l) => l.match(/^([A-Z0-9_]+)=(.+)$/)).filter(Boolean).map((m) => [m[1], m[2].trim()]));
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, locale: "zh-CN" });
const page = await ctx.newPage();
await page.goto(BASE + "/login/", { waitUntil: "networkidle" });
await page.getByPlaceholder("请输入账号").fill(env.ADMIN_USER);
await page.getByPlaceholder("请输入密码").fill(env.ADMIN_PASSWORD);
const c = page.getByRole("checkbox", { name: /我已阅读并同意/ });
if (await c.count()) await c.check();
await page.getByRole("button", { name: "登录", exact: true }).click();
await page.getByRole("heading", { name: "哪里不舒服？" }).waitFor();
const offenders = await page.evaluate(() => {
  const out = [];
  document.querySelectorAll("*").forEach((el) => {
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.right > window.innerWidth + 1) {
      out.push(el.tagName + "." + String(el.className).slice(0, 60) + " right=" + Math.round(r.right) + " w=" + Math.round(r.width));
    }
  });
  return out.slice(0, 12);
});
console.log(offenders.join("\n"));
await browser.close();
