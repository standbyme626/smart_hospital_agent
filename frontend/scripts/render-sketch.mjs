import { chromium } from "@playwright/test";
import fs from "node:fs";

const svg = fs.readFileSync("/home/kkk/Project/smart_hospital_agent_v2/docs/assets/v34-layout-sketch.svg", "utf-8");
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1240, height: 1660 }, deviceScaleFactor: 2 });
await page.setContent(
  "<!doctype html><html><head><meta charset='utf-8'><style>html,body{margin:0;background:#fff}svg{display:block}</style></head><body>" +
    svg +
    "</body></html>",
  { waitUntil: "load" },
);
await page.waitForTimeout(300);
await page.screenshot({
  path: "/home/kkk/Project/smart_hospital_agent_v2/docs/assets/v34-layout-sketch.png",
  clip: { x: 0, y: 0, width: 1240, height: 1660 },
});
console.log("png saved");
await browser.close();
