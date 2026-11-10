#!/usr/bin/env bash
# Build and deploy the real frontend to 106 (production root).
# Usage: scripts/deploy_frontend.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/frontend"

# Same-origin API: nginx proxies /api/ to the tunnel. Never bake localhost here.
NEXT_PUBLIC_API_BASE="" npm run build

STAMP="$(date +%Y%m%d-%H%M%S)"
timeout 20 ssh -o BatchMode=yes tx106-friend "cp -a /var/www/hospital /var/www/hospital.bak-$STAMP"
rsync -az --delete --exclude preview "$ROOT/frontend/out/" tx106-friend:/var/www/hospital/
timeout 20 ssh -o BatchMode=yes tx106-friend 'find /var/www/hospital -maxdepth 1 -type d -exec chmod 755 {} \; ; find /var/www/hospital -type f -exec chmod 644 {} \;'
echo "已发布：https://hospital.kk.zhzepei.cn/（备份 /var/www/hospital.bak-$STAMP）"
