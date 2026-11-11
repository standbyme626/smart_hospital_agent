import { chromium } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const BASE = "https://hospital.kk.zhzepei.cn";
const OUT = "/tmp/prod-shots";
fs.mkdirSync(OUT, { recursive: true });

const env = Object.fromEntries(
  fs
    .readFileSync("/home/kkk/Project/smart_hospital_agent_v2/.env", "utf-8")
    .split("\n")
    .map((line) => line.match(/^([A-Z0-9_]+)=(.+)$/))
    .filter(Boolean)
    .map((m) => [m[1], m[2].trim()]),
);

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, locale: "zh-CN" });
const page = await context.newPage();

const bad401 = [];
page.on("response", (resp) => {
  if (resp.status() === 401) bad401.push(resp.url());
});
const pageErrors = [];
page.on("pageerror", (err) => pageErrors.push(String(err).slice(0, 160)));

await page.goto(BASE + "/login/", { waitUntil: "networkidle" });
await page.getByPlaceholder("请输入账号").fill(env.ADMIN_USER);
await page.getByPlaceholder("请输入密码").fill(env.ADMIN_PASSWORD);
const consent = page.getByRole("checkbox", { name: /我已阅读并同意/ });
if (await consent.count()) await consent.check();
await page.getByRole("button", { name: "登录", exact: true }).click();
await page.getByText("哪里不舒服？").waitFor({ timeout: 30000 });
console.log("登录成功，当前 URL:", page.url());

const token = await page.evaluate(() => window.localStorage.getItem("sha_token"));
console.log("token 长度:", token ? token.length : 0);

const paths = [
  ["/", "患者端首页"],
  ["/messages/", "消息中心"],
  ["/clinician/", "医生端"],
  ["/clinician/print/", "文书打印"],
  ["/ops/", "运营端"],
  ["/ops/reviews/", "支付异常"],
  ["/platform/", "管理端总览"],
  ["/platform/config/", "配置管理"],
  ["/platform/knowledge/", "知识库"],
  ["/platform/integrations/", "接入与实验"],
  ["/admin/", "旧后台"],
];

for (const [route, label] of paths) {
  const resp = await page.goto(BASE + route, { waitUntil: "networkidle" });
  const body = (await page.locator("body").innerText()).slice(0, 50).replace(/\s+/g, " ");
  const loggedOut = page.url().includes("/login");
  console.log(resp.status() + " " + route + " " + label + " | 掉登录:" + loggedOut + " | " + body);
  if (route === "/" || route === "/ops/" || route === "/platform/knowledge/") {
    await page.screenshot({ path: path.join(OUT, "desktop" + route.replace(/\//g, "_") + ".png"), fullPage: false });
  }
}

const mobile = await browser.newContext({ viewport: { width: 390, height: 844 }, locale: "zh-CN" });
const mp = await mobile.newPage();
await mp.addInitScript((tk) => window.localStorage.setItem("sha_token", tk), token);
for (const [route, label] of [["/", "移动首页"], ["/messages/", "移动消息"]]) {
  await mp.goto(BASE + route, { waitUntil: "networkidle" });
  console.log("mobile " + route + " " + label + " | 掉登录:" + mp.url().includes("/login"));
  await mp.screenshot({ path: path.join(OUT, "mobile" + route.replace(/\//g, "_") + ".png"), fullPage: false });
}

console.log("401 响应数:", bad401.length, bad401.slice(0, 3));
console.log("页面 JS 错误:", pageErrors.length, pageErrors.slice(0, 2));
await browser.close();
