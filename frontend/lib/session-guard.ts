export function memberMismatch(
  sessionMemberId: string | null | undefined,
  activeMemberId: string | null | undefined,
): boolean {
  return Boolean(sessionMemberId) && sessionMemberId !== activeMemberId;
}
