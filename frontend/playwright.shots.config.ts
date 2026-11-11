import base from "./playwright.config";

/** 只跑截图脚本：node -e 不会命中，日常 E2E 也不该为它多花 30 秒。
 *  用法：npx playwright test -c playwright.shots.config.ts
 */
export default {
  ...base,
  testMatch: ["e2e/shots.manual.ts"],
  testDir: "./e2e",
};
