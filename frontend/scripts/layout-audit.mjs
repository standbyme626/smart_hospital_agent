import { chromium } from "@playwright/test";
import fs from "node:fs";

const BASE = "https://hospital.kk.zhzepei.cn";
const env = Object.fromEntries(
  fs.readFileSync("/home/kkk/Project/smart_hospital_agent_v2/.env", "utf-8")
    .split("\n").map((l) => l.match(/^([A-Z0-9_]+)=(.+)$/)).filter(Boolean)
    .map((m) => [m[1], m[2].trim()]),
);

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, locale: "zh-CN" });
const page = await ctx.newPage();
await page.goto(BASE + "/login/", { waitUntil: "networkidle" });
await page.getByPlaceholder("请输入账号").fill(env.ADMIN_USER);
await page.getByPlaceholder("请输入密码").fill(env.ADMIN_PASSWORD);
const consent = page.getByRole("checkbox", { name: /我已阅读并同意/ });
if (await consent.count()) await consent.check();
await page.getByRole("button", { name: "登录", exact: true }).click();
await page.getByText("哪里不舒服？").waitFor({ timeout: 30000 });

const routes = ["/", "/messages/", "/clinician/", "/clinician/print/", "/ops/", "/ops/reviews/", "/platform/", "/platform/config/", "/platform/knowledge/", "/entries/"];

for (const route of routes) {
  await page.goto(BASE + route, { waitUntil: "networkidle" });
  const m = await page.evaluate(() => {
    const rect = (el) => {
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return { x: Math.round(r.x), w: Math.round(r.width), h: Math.round(r.height), y: Math.round(r.y) };
    };
    const header = document.querySelector("header");
    const mainEl = document.querySelector("main");
    const h1 = document.querySelector("h1");
    const firstCard = document.querySelector(".card");
    const navs = Array.from(document.querySelectorAll("nav")).map((n) => {
      const r = n.getBoundingClientRect();
      return { y: Math.round(r.y), h: Math.round(r.height), cls: n.className.slice(0, 40) };
    });
    const cs = firstCard ? getComputedStyle(firstCard) : null;
    return {
      header: rect(header),
      main: rect(mainEl),
      h1Size: h1 ? getComputedStyle(h1).fontSize : null,
      h1Text: h1 ? h1.innerText.slice(0, 14) : null,
      cardRadius: cs ? cs.borderRadius : null,
      cardPadding: cs ? cs.padding : null,
      navs,
      bg: getComputedStyle(document.body).backgroundColor,
    };
  });
  const navInfo = m.navs.map((n) => "nav@" + n.y + "px/" + n.h + "px").join(" ");
  console.log(route.padEnd(22) +
    " header:" + (m.header ? m.header.h + "px" : "—").padEnd(8) +
    " main:" + (m.main ? m.main.x + "/" + m.main.w : "—").padEnd(12) +
    " h1:" + (m.h1Size || "—").padEnd(6) +
    " cardR:" + (m.cardRadius || "—").padEnd(8) +
    " nav:" + navInfo.padEnd(26) +
    " 标题:" + (m.h1Text || "—"));
}
await browser.close();
