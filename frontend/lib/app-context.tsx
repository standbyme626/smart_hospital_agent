"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import {
  UnauthorizedError,
  clearToken,
  getFamily,
  getMe,
  login as apiLogin,
  track,
} from "./api";
import type { Member, User } from "./types";

type FontScale = "md" | "lg";
type Contrast = "normal" | "high";
type Motion = "normal" | "reduced";

type SettingsState = {
  fontScale: FontScale;
  contrast: Contrast;
  motion: Motion;
};

type AppContextValue = {
  user: User | null;
  authReady: boolean;
  members: Member[];
  activeMember: Member | null;
  activeMemberId: string | null;
  setActiveMemberId: (id: string) => void;
  refreshFamily: () => Promise<void>;
  login: (username: string, password: string) => Promise<string>;
  logout: () => void;
  settings: SettingsState;
  setSettings: (next: Partial<SettingsState>) => void;
  toast: string | null;
  showToast: (message: string) => void;
};

const AppContext = createContext<AppContextValue | null>(null);
const ACTIVE_MEMBER_KEY = "sha_active_member";
const SETTINGS_KEY = "sha_settings";

const defaultSettings: SettingsState = { fontScale: "md", contrast: "normal", motion: "normal" };

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [members, setMembers] = useState<Member[]>([]);
  const [activeMemberId, setActiveMemberIdState] = useState<string | null>(null);
  const [settings, setSettingsState] = useState<SettingsState>(defaultSettings);
  const [toast, setToast] = useState<string | null>(null);

  const applySettings = useCallback((next: SettingsState) => {
    const root = document.documentElement;
    root.dataset.font = next.fontScale;
    root.dataset.contrast = next.contrast;
    root.dataset.motion = next.motion;
    window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(next));
  }, []);

  useEffect(() => {
    try {
      const savedSettings = window.localStorage.getItem(SETTINGS_KEY);
      if (savedSettings) {
        const parsed = { ...defaultSettings, ...JSON.parse(savedSettings) } as SettingsState;
        setSettingsState(parsed);
        applySettings(parsed);
      } else {
        applySettings(defaultSettings);
      }
    } catch {
      applySettings(defaultSettings);
    }
    try {
      const savedMember = window.localStorage.getItem(ACTIVE_MEMBER_KEY);
      if (savedMember) setActiveMemberIdState(savedMember);
    } catch {
      // ignore
    }
  }, [applySettings]);

  const refreshFamily = useCallback(async () => {
    try {
      const list = await getFamily();
      setMembers(list);
      const saved = window.localStorage.getItem(ACTIVE_MEMBER_KEY);
      const preferred = list.find((item) => item.id === saved) || list.find((item) => item.is_self) || list[0];
      if (preferred) {
        setActiveMemberIdState(preferred.id);
        window.localStorage.setItem(ACTIVE_MEMBER_KEY, preferred.id);
      }
    } catch (err) {
      if (err instanceof UnauthorizedError) setUser(null);
    }
  }, []);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const me = await getMe();
        if (!alive) return;
        setUser(me);
        await refreshFamily();
      } catch (err) {
        if (alive && err instanceof UnauthorizedError) setUser(null);
      } finally {
        if (alive) setAuthReady(true);
      }
    })();
    return () => {
      alive = false;
    };
  }, [refreshFamily]);

  const setActiveMemberId = useCallback((id: string) => {
    setActiveMemberIdState(id);
    window.localStorage.setItem(ACTIVE_MEMBER_KEY, id);
    track("family_switch", { member_id: id });
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    const result = await apiLogin(username, password);
    const me = await getMe();
    setUser(me);
    await refreshFamily();
    return result.role;
  }, [refreshFamily]);

  const logout = useCallback(() => {
    clearToken();
    setUser(null);
    setMembers([]);
    setActiveMemberIdState(null);
    window.localStorage.removeItem(ACTIVE_MEMBER_KEY);
    window.location.href = "/login/";
  }, []);

  const setSettings = useCallback(
    (next: Partial<SettingsState>) => {
      setSettingsState((prev) => {
        const merged = { ...prev, ...next };
        applySettings(merged);
        return merged;
      });
    },
    [applySettings],
  );

  const showToast = useCallback((message: string) => {
    setToast(message);
    window.setTimeout(() => setToast(null), 3200);
  }, []);

  const activeMember = useMemo(
    () => members.find((item) => item.id === activeMemberId) || members.find((item) => item.is_self) || members[0] || null,
    [members, activeMemberId],
  );

  const value = useMemo<AppContextValue>(
    () => ({
      user,
      authReady,
      members,
      activeMember,
      activeMemberId,
      setActiveMemberId,
      refreshFamily,
      login,
      logout,
      settings,
      setSettings,
      toast,
      showToast,
    }),
    [
      user,
      authReady,
      members,
      activeMember,
      activeMemberId,
      setActiveMemberId,
      refreshFamily,
      login,
      logout,
      settings,
      setSettings,
      toast,
      showToast,
    ],
  );

  return (
    <AppContext.Provider value={value}>
      {children}
      {toast ? (
        <div className="fixed bottom-24 left-1/2 z-[80] -translate-x-1/2 rounded-full bg-ink px-4 py-2 text-sm text-white shadow-lg lg:bottom-8">
          {toast}
        </div>
      ) : null}
    </AppContext.Provider>
  );
}

export function useApp(): AppContextValue {
  const context = useContext(AppContext);
  if (!context) throw new Error("useApp must be used inside AppProvider");
  return context;
}
