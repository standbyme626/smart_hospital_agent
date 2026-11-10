#!/usr/bin/env bash
# Deploy the clickable prototype to the public preview path.
# Usage: scripts/deploy_preview.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/prototype"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

cp "$SRC/dual-end.html" "$BUILD/dual-end.html"
cp "$SRC/index.html" "$BUILD/index.html"

python3 - "$BUILD" <<'PY'
import pathlib, sys
b = pathlib.Path(sys.argv[1])
banner = (
    '<div style="position:fixed;left:0;right:0;top:0;z-index:10000;background:#8A5310;color:#fff;'
    'font:600 12.5px/1.6 \'Noto Sans CJK SC\',sans-serif;padding:8px 16px;text-align:center">'
    '演示环境 · 全部数据为模拟，不产生真实挂号 / 支付 / 通知 · '
    '<a href="/" style="color:#FFE7B8;text-decoration:underline">返回真实站点</a></div>'
    '<style>body{padding-top:38px}</style>'
)
p = b / 'dual-end.html'
s = p.read_text(encoding='utf-8')
s = s.replace('<meta charset="utf-8">', '<meta charset="utf-8">\n<meta name="robots" content="noindex,nofollow">', 1)
s = s.replace('<body>', '<body>\n' + banner, 1)
p.write_text(s, encoding='utf-8')
i = b / 'index.html'
t = i.read_text(encoding='utf-8')
t = t.replace('<meta charset="utf-8">', '<meta charset="utf-8">\n<meta name="robots" content="noindex,nofollow">', 1)
i.write_text(t, encoding='utf-8')
PY

rsync -az --delete "$BUILD"/ tx106-friend:/var/www/hospital/preview/
timeout 20 ssh -o BatchMode=yes tx106-friend 'chmod 755 /var/www/hospital/preview && chmod 644 /var/www/hospital/preview/*'
echo "已部署：https://hospital.kk.zhzepei.cn/preview/"
