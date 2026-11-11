import { chromium } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const BASE = "https://hospital.kk.zhzepei.cn";
const OUT = "/tmp/prod-shots2";
fs.mkdirSync(OUT, { recursive: true });
const env = Object.fromEntries(
  fs.readFileSync("/home/kkk/Project/smart_hospital_agent_v2/.env", "utf-8")
    .split("\n").map((l) => l.match(/^([A-Z0-9_]+)=(.+)$/)).filter(Boolean)
    .map((m) => [m[1], m[2].trim()]),
);

const browser = await chromium.launch();
const desktop = await browser.newContext({ viewport: { width: 1440, height: 900 }, locale: "zh-CN" });
const page = await desktop.newPage();
await page.goto(BASE + "/login/", { waitUntil: "networkidle" });
await page.getByPlaceholder("请输入账号").fill(env.ADMIN_USER);
await page.getByPlaceholder("请输入密码").fill(env.ADMIN_PASSWORD);
const consent = page.getByRole("checkbox", { name: /我已阅读并同意/ });
if (await consent.count()) await consent.check();
await page.getByRole("button", { name: "登录", exact: true }).click();
await page.getByText("哪里不舒服？").waitFor({ timeout: 30000 });
const token = await page.evaluate(() => window.localStorage.getItem("sha_token"));

const paths = ["/", "/messages/", "/clinician/", "/ops/", "/platform/", "/platform/config/", "/platform/knowledge/"];
for (const route of paths) {
  await page.goto(BASE + route, { waitUntil: "networkidle" });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  console.log("desktop " + route + " 横向溢出:" + overflow + "px");
  await page.screenshot({ path: path.join(OUT, "d" + route.replace(/\//g, "_") + ".png") });
}

const mob = await browser.newContext({ viewport: { width: 390, height: 844 }, locale: "zh-CN" });
const mp = await mob.newPage();
await mp.addInitScript((tk) => window.localStorage.setItem("sha_token", tk), token);
for (const route of ["/", "/messages/", "/clinician/", "/ops/", "/platform/"]) {
  await mp.goto(BASE + route, { waitUntil: "networkidle" });
  const overflow = await mp.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  const navCount = await mp.locator("nav.fixed a").count();
  console.log("mobile " + route + " 横向溢出:" + overflow + "px 底栏项:" + navCount);
  await mp.screenshot({ path: path.join(OUT, "m" + route.replace(/\//g, "_") + ".png") });
}
await browser.close();
