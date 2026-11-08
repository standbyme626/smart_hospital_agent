export type RestorableOrder = { status: string };
export type StoredLock = {
  lock_id: string;
  slot_id: string;
  expires_in: number;
  expires_at?: string;
  member_id?: string | null;
};

export function pickRestorableOrder<T extends RestorableOrder>(orders: T[]): T | null {
  return orders.find((item) => item.status !== "cancelled") ?? null;
}

export function parseStoredLock(
  raw: string | null,
  memberId: string | null | undefined,
  now: number = Date.now(),
): StoredLock | null {
  if (!raw) return null;
  try {
    const saved = JSON.parse(raw) as StoredLock;
    if ((saved.member_id || null) !== (memberId || null)) return null;
    if (saved.expires_at && new Date(saved.expires_at).getTime() <= now) return null;
    return saved;
  } catch {
    return null;
  }
}

export function serializeStoredLock(lock: StoredLock, memberId: string | null | undefined): string {
  return JSON.stringify({ ...lock, member_id: memberId || null });
}
