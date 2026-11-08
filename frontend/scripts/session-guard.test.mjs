import assert from "node:assert/strict";
import test from "node:test";

import { memberMismatch } from "../lib/session-guard.ts";

test("no session member never mismatches", () => {
  assert.equal(memberMismatch(null, "mem_1"), false);
  assert.equal(memberMismatch(undefined, undefined), false);
  assert.equal(memberMismatch("", "mem_1"), false);
});

test("missing active member with known session member is a mismatch", () => {
  assert.equal(memberMismatch("mem_1", null), true);
  assert.equal(memberMismatch("mem_1", undefined), true);
});

test("same member matches, different member mismatches", () => {
  assert.equal(memberMismatch("mem_1", "mem_1"), false);
  assert.equal(memberMismatch("mem_1", "mem_2"), true);
});
