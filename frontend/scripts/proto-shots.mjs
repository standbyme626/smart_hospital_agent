import { chromium } from "@playwright/test";
import fs from "node:fs";

const OUT = "/tmp/proto-shots";
fs.mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, locale: "zh-CN" });
await page.goto("file:///home/kkk/Project/smart_hospital_agent_v2/prototype/dual-end.html", { waitUntil: "load" });
await page.waitForTimeout(400);

const screens = [
  ["patient", "p-home", "患者端首页"],
  ["doctor", "d-queue", "医生端队列"],
  ["ops", "o-review", "运营端支付审核"],
  ["admin", "a-overview", "管理端总览"],
];

for (const [end, screen, label] of screens) {
  await page.evaluate(([e, s]) => {
    document.getElementById("entry").classList.remove("on");
    document.getElementById("login").classList.remove("on");
    window.switchEnd(e);
    window.showScreen(s);
  }, [end, screen]);
  await page.waitForTimeout(500);
  const size = await page.evaluate(() => {
    const rect = (el) => {
      if (!el) return null;
      const b = el.getBoundingClientRect();
      return Math.round(b.x) + "/" + Math.round(b.width) + "x" + Math.round(b.height);
    };
    const app = document.querySelector(".app");
    return {
      top: rect(document.querySelector(".topbar")),
      grid: app ? getComputedStyle(app).gridTemplateColumns : null,
      main: rect(document.querySelector(".main")),
      h1: document.querySelector(".screen.on h1") ? getComputedStyle(document.querySelector(".screen.on h1")).fontSize : null,
      title: document.querySelector(".screen.on h1") ? document.querySelector(".screen.on h1").innerText.slice(0, 12) : null,
    };
  });
  console.log(label + " | topbar " + size.top + " | grid " + size.grid + " | main " + size.main + " | h1 " + size.h1 + " | " + size.title);
  await page.screenshot({ path: OUT + "/" + screen + ".png" });
}
await browser.close();
