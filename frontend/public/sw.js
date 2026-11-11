// 最小 service worker：让 PWA 可安装；离线缓存后续接入。
self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", () => {
  // 透传：不缓存，避免脏数据。
});
