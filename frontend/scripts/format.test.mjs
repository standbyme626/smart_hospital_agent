import assert from "node:assert/strict";
import test from "node:test";

import {
  abnormalStatusMeta,
  ageFromBirthDate,
  formatSessionTitle,
  severityMeta,
  stripMemberContext,
  timelineHref,
} from "../lib/format.ts";

function expectedAge(year, month, day) {
  const now = new Date();
  let age = now.getFullYear() - year;
  if (now.getMonth() + 1 < month || (now.getMonth() + 1 === month && now.getDate() < day)) age -= 1;
  return age;
}

test("ageFromBirthDate computes age from a YYYY-MM-DD birth date", () => {
  assert.equal(ageFromBirthDate("1994-11-09"), expectedAge(1994, 11, 9));
  assert.equal(ageFromBirthDate("2000-01-01"), expectedAge(2000, 1, 1));
  assert.equal(ageFromBirthDate(" 1994-11-09 "), expectedAge(1994, 11, 9));
});

test("ageFromBirthDate rejects malformed, impossible, future and >120 dates", () => {
  assert.equal(ageFromBirthDate(null), null);
  assert.equal(ageFromBirthDate(undefined), null);
  assert.equal(ageFromBirthDate(""), null);
  assert.equal(ageFromBirthDate("1994-11-9"), null);
  assert.equal(ageFromBirthDate("1994-13-01"), null);
  assert.equal(ageFromBirthDate("1994-02-30"), null);
  assert.equal(ageFromBirthDate("not-a-date"), null);
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  const future = `${tomorrow.getFullYear()}-${String(tomorrow.getMonth() + 1).padStart(2, "0")}-${String(tomorrow.getDate()).padStart(2, "0")}`;
  assert.equal(ageFromBirthDate(future), null);
  assert.equal(ageFromBirthDate(`${new Date().getFullYear() - 121}-01-01`), null);
});

test("formatSessionTitle turns a legacy member-context title into a symptom summary", () => {
  const legacy = [
    "就诊人：张三（self）",
    "性别：男",
    "年龄：32",
    "出生日期：1994-11-09",
    "",
    "主要部位：头面部",
    "症状：疼痛、发热",
    "持续时间：1–3 天",
  ].join("\n");
  assert.equal(formatSessionTitle(legacy), "头面部 · 疼痛、发热 · 1–3 天");
});

test("formatSessionTitle takes content after the last 用户输入 marker", () => {
  const stored = [
    "就诊人：张三（self）；性别：男",
    "",
    "用户输入：就诊人：张三（self）",
    "性别：男",
    "主要部位：头面部",
    "症状：疼痛",
  ].join("\n");
  assert.equal(formatSessionTitle(stored), "头面部 · 疼痛");
});

test("formatSessionTitle keeps clean titles and falls back when empty", () => {
  assert.equal(formatSessionTitle("头面部 · 疼痛 · 1–3 天"), "头面部 · 疼痛 · 1–3 天");
  assert.equal(formatSessionTitle(null), "上次问诊");
  assert.equal(formatSessionTitle(""), "上次问诊");
  assert.equal(formatSessionTitle("就诊人：张三（self）\n性别：男\n年龄：32"), "上次问诊");
  assert.equal(formatSessionTitle("   ", "兜底"), "兜底");
});

test("formatSessionTitle truncates long summaries", () => {
  const long = "补充描述：" + "头痛发热".repeat(30);
  const formatted = formatSessionTitle(long);
  assert.equal(formatted.length, 61);
  assert.ok(formatted.endsWith("…"));
});

test("stripMemberContext drops member lines and keeps user symptom lines", () => {
  const message = [
    "就诊人：张三（self）",
    "性别：男",
    "年龄：32",
    "档案过敏史：青霉素",
    "档案既往史：高血压",
    "",
    "主要部位：头面部",
    "症状：疼痛",
    "过敏史：无",
  ].join("\n");
  assert.equal(stripMemberContext(message), "主要部位：头面部\n症状：疼痛\n过敏史：无");
});

test("stripMemberContext prefers text after the last 用户输入 marker", () => {
  const stored = "就诊人：张三（self）\n\n用户输入：主要部位：头面部\n症状：疼痛";
  assert.equal(stripMemberContext(stored), "主要部位：头面部\n症状：疼痛");
  assert.equal(stripMemberContext(null), "");
});

test("stripMemberContext leaves free-text follow-ups untouched", () => {
  assert.equal(stripMemberContext("年龄：35"), "年龄：35");
  assert.equal(stripMemberContext("过敏史：无"), "过敏史：无");
  assert.equal(stripMemberContext("补充描述：这两天晚上更明显"), "补充描述：这两天晚上更明显");
});

test("formatSessionTitle trims trailing separators and short fragments", () => {
  const legacyTruncated = "就诊人：张三（self）\n性别：男\n主要部位：头面部\n症状：疼、";
  assert.equal(formatSessionTitle(legacyTruncated), "头面部 · 疼");
  assert.equal(formatSessionTitle("症状：痛"), "上次问诊");
  assert.equal(formatSessionTitle("主要部位：头面部，"), "头面部");
});

test("timelineHref deep-links reports, consults and orders", () => {
  assert.equal(timelineHref("report", { report_id: "r1" }), "/reports/?id=r1");
  assert.equal(timelineHref("consult", { session_id: "s 1" }), "/consult/?session=s%201");
  assert.equal(timelineHref("booking", { order_id: "o1" }), "/services/?order=o1");
  assert.equal(timelineHref("followup", { summary: "x" }), null);
  assert.equal(timelineHref("report", null), null);
});

test("abnormalStatusMeta only colors known states", () => {
  assert.equal(abnormalStatusMeta("high").label, "偏高");
  assert.equal(abnormalStatusMeta("HIGH").label, "偏高");
  assert.equal(abnormalStatusMeta("low").label, "偏低");
  assert.equal(abnormalStatusMeta("normal").label, "正常");
  const unknown = abnormalStatusMeta("pending");
  assert.equal(unknown.label, "未知");
  assert.equal(unknown.key, "unknown");
  assert.equal(abnormalStatusMeta(null).label, "未知");
});

test("severityMeta maps drug risk levels to Chinese", () => {
  assert.equal(severityMeta("contraindicated").label, "禁用 / 禁忌");
  assert.equal(severityMeta("major").label, "严重");
  assert.equal(severityMeta("moderate").label, "中等");
  assert.equal(severityMeta("minor").label, "轻微");
  assert.equal(severityMeta("none").label, "未发现已知相互作用");
  assert.equal(severityMeta("weird").label, "未知");
  assert.ok(severityMeta("major").dangerous);
});
