import assert from "node:assert/strict";
import test from "node:test";

import {
  parseStoredLock,
  pickRestorableOrder,
  serializeStoredLock,
} from "../lib/booking-restore.ts";

test("picks the newest non-cancelled order", () => {
  assert.equal(pickRestorableOrder([]), null);
  const orders = [
    { id: "a", status: "cancelled" },
    { id: "b", status: "pending_review" },
    { id: "c", status: "created" },
  ];
  assert.equal(pickRestorableOrder(orders)?.id, "b");
  assert.equal(pickRestorableOrder([{ id: "x", status: "cancelled" }]), null);
});

test("lock storage round-trips and is scoped to the member", () => {
  const lock = {
    lock_id: "lock_1",
    slot_id: "slot_1",
    expires_in: 600,
    expires_at: "2030-01-01T00:00:00+00:00",
  };
  const raw = serializeStoredLock(lock, "mem_1");
  assert.equal(parseStoredLock(raw, "mem_1")?.lock_id, "lock_1");
  assert.equal(parseStoredLock(raw, "mem_2"), null);
  assert.equal(parseStoredLock(raw, null), null);
});

test("expired or malformed stored locks are ignored", () => {
  const expired = JSON.stringify({
    lock_id: "lock_old",
    slot_id: "slot_1",
    expires_in: 600,
    expires_at: "2020-01-01T00:00:00+00:00",
    member_id: "mem_1",
  });
  assert.equal(parseStoredLock(expired, "mem_1"), null);
  assert.equal(parseStoredLock("not-json", "mem_1"), null);
  assert.equal(parseStoredLock(null, "mem_1"), null);
});

test("locks without member ids only match memberless sessions", () => {
  const legacy = JSON.stringify({ lock_id: "lock_l", slot_id: "slot_1", expires_in: 600 });
  assert.equal(parseStoredLock(legacy, null)?.lock_id, "lock_l");
  assert.equal(parseStoredLock(legacy, "mem_1"), null);
});
