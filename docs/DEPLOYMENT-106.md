# 106 部署记录（hospital.kk.zhzepei.cn）

> 2026-11-08 上线 v3.0。静态前端 + 家服务器 API + 反向 SSH 隧道 + Let's Encrypt。
> **v3.1（2026-11-09）**：代码在本地分支 `v3.1` 完成，**尚未部署**；部署顺序见第 2 节。线上仍是 v3.0 旧进程。
> 本文不含任何密码/密钥；凭据在服务器 `/root/.credentials/smart-hospital.env`（600）。v3.1 部署不改凭据文件内容（轮换需主人单独批准）。

## 1. 拓扑

```text
浏览器
  │ https://hospital.kk.zhzepei.cn
  ▼
106.55.16.53 (tx106-friend) · nginx 443
  ├─ /      → /var/www/hospital  (frontend/out 静态导出)
  └─ /api/  → 127.0.0.1:18087
                    │ 反向 SSH 隧道（家 → 106）
                    ▼
家服务器 127.0.0.1:8020 · FastAPI
  ├─ PostgreSQL 16 + pgvector :5434
  ├─ mock 医院 MCP :9100
  └─ WinPC Ollama 隧道 :11434
```

DNS：`*.kk.zhzepei.cn` 已解析到 `106.55.16.53`，`hospital.kk.zhzepei.cn` 无需新增记录。

## 2. v3.1 部署顺序（部署时执行，当前未执行）

顺序固定为：**停止写入 → 备份 → 代码切换 → 启动 API（自动幂等迁移）→ 前端构建/发布 → nginx 示例审核（可选安装）→ Kuma 验证**。任一步失败先停手，按第 9 节回滚。

### 2.1 备份（先停 API，保证 dump 一致）

```bash
systemctl stop smart-hospital-api.service
cd /home/kkk/Project/smart_hospital_agent_v2
bash scripts/backup.sh
ls -l /home/data/backups/smart-hospital/
```

验证产物可读：

```bash
docker exec -i sha_v2_db sh -c 'pg_restore -l' \
  < /home/data/backups/smart-hospital/db-<备份时间戳>.dump | head
gzip -t /home/data/backups/smart-hospital/uploads-<备份时间戳>.tar.gz
```

备份目录 700 / 文件 600，保留 7 天；timer `smart-hospital-backup.timer` 每天 03:30 自动跑。备份含患者数据，禁止复制到公网。

### 2.2 代码切换

```bash
cd /home/kkk/Project/smart_hospital_agent_v2
git status --short          # 应为空
git rev-parse --short HEAD  # 记录部署 commit
```

本次只在本地 `v3.1` 分支提交，未 push；部署时以该工作区/发布包为准，记录 commit 以便回滚。

### 2.3 启动 API：自动迁移（幂等）

```bash
systemctl start smart-hospital-api.service
journalctl -u smart-hospital-api.service -n 50 --no-pager
curl -fsS http://127.0.0.1:8020/api/health
```

- FastAPI lifespan 中 `db.init_schema()` 自动执行全部迁移，**不需要手工 SQL**。
- 迁移幂等：建表 `CREATE TABLE IF NOT EXISTS`；加列 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`；索引 `CREATE INDEX IF NOT EXISTS`；回填 `UPDATE ... WHERE ... IS NULL`。重复启动/重试安全。
- v3.0 数据保留：只增表/列，不删旧列；`bookings.user_id` 为空时先按 `bookings.session_id = sessions.id` 回填 `sessions.user_id`，无匹配会话的旧行再回退到 `session_id`。
- `bootstrap_users` 仅在 `users` 表为空且 `app_meta` 无 bootstrap marker 时创建 `ADMIN_USER` / `PATIENT_USER`；不会覆盖已有用户，账号删除后也不会自动重建。已有用户改密码用 `cd backend && .venv/bin/python -m scripts.set_password <username>`。
- **首次启动 v3.1 API 前必须先处理初始密码**：bootstrap 读取环境变量 `ADMIN_PASSWORD` / `PATIENT_PASSWORD`。若尚未启动过，先把新的强密码写入 600 权限的 `.env` 再启动；若已经用弱密码启动过，重建账号需先删除 `users` 中的账号并删除 `app_meta` 的 `users_bootstrapped` 标记（或直接改密码），否则弱密码会保留。**本任务不轮换线上凭据**，以上操作需主人批准后执行。
- 若日志出现 `[startup] database init failed`，不要继续前端发布；先查 DB 连接/权限，修复后重启 API（迁移可重跑）。

### 2.4 前端构建与发布

```bash
# 106 先备份上一版静态文件
ssh tx106-friend 'cp -a /var/www/hospital /var/www/hospital.bak-$(date +%Y%m%d-%H%M%S)'

cd /home/kkk/Project/smart_hospital_agent_v2/frontend
NEXT_PUBLIC_API_BASE= npm run build
ssh tx106-friend "mkdir -p /var/www/hospital"
scp -r out/* tx106-friend:/var/www/hospital/
```

`NEXT_PUBLIC_API_BASE=` 留空 = 同源 `/api`，由 nginx 反代。保留 `/var/www/hospital.bak-*` 用于回滚。

### 2.5 nginx 示例审核（不默认安装）

`deploy/nginx-hospital-v31.conf` 是**示例，未在 106 安装**。部署时：

```bash
# 只读对比（不要直接覆盖线上文件）
ssh tx106-friend 'cat /etc/nginx/sites-available/hospital.kk.zhzepei.cn' \
  > /tmp/live-nginx-hospital.conf
diff -u /tmp/live-nginx-hospital.conf deploy/nginx-hospital-v31.conf || true
```

确认差异后再：备份线上文件 → 安装新配置 → `nginx -t` → `systemctl reload nginx`。没有主人批准，不安装、不 reload。

**Basic Auth 边界（必须遵守）**：

- Basic Auth **只加在 `location ^~ /admin` 静态页**；htpasswd 文件真身 `/root/.credentials/smart-hospital-admin-htpasswd`（600），nginx worker 读不了 `/root`，安装 640 `root:www-data` 副本到 `/etc/nginx/htpasswd/smart-hospital-admin`。
- **绝对不要给 `/api/v1/admin/` 加 `auth_basic`**：nginx 会消费浏览器同一个 `Authorization` 头，Bearer 请求全部 401。管理 API 的权限由应用 `require_admin`（JWT + `role=admin`）负责。
- 需要网络层加固时，用 Cloudflare Access 或 IP allowlist（`allow <CIDR>; deny all;`），不要改写/吞掉 `Authorization`。
- 不把 `$http_authorization` / `$request_body` 加进 access log。

### 2.6 验证与 Kuma

```bash
curl -fsS https://hospital.kk.zhzepei.cn/api/health   # {"ok":true,"version":"3.1.0",...}
curl -I https://hospital.kk.zhzepei.cn/
```

- Kuma（tx43）监控项 `Smart Hospital API health` → `https://hospital.kk.zhzepei.cn/api/health`，期望 200 且 `{"ok": true}`；M0 记录为暂停；**部署成功后确认并启用**（若主人已手动启用则跳过）。
- 管理入口：`/admin` 未带 Basic → 401（仅在 nginx v3.1 配置安装后）；`/api/v1/admin/users` 未带 JWT → 401，患者 JWT → 403。
- 按 `docs/V3.1-RELEASE.md` 的验收清单跑患者问诊、报告、支付审核、成员隔离。

## 3. 构建与发布静态前端（快速参考）

> v3.1 部署请按 2.4 先备份上一版 `/var/www/hospital`；本节只是日常重建命令。

```bash
cd /home/kkk/Project/smart_hospital_agent_v2/frontend
NEXT_PUBLIC_API_BASE= npm run build
ssh tx106-friend "mkdir -p /var/www/hospital"
scp -r out/* tx106-friend:/var/www/hospital/
```

`NEXT_PUBLIC_API_BASE=` 留空表示同源 `/api`，由 nginx 反代。开发环境用 `frontend/.env.local` 指向 `http://127.0.0.1:8020`。

## 4. 家服务器 systemd

### 4.1 API

`/etc/systemd/system/smart-hospital-api.service`

```ini
[Unit]
Description=Smart Hospital Agent API
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/kkk/Project/smart_hospital_agent_v2/backend
Environment=PYTHONPATH=/home/kkk/Project/smart_hospital_agent_v2/backend
# Optional: pydantic-settings already loads repo .env; keep both in sync if added.
EnvironmentFile=-/home/kkk/Project/smart_hospital_agent_v2/.env
ExecStart=/home/kkk/Project/smart_hospital_agent_v2/backend/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8020
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### 4.2 Mock 医院 MCP

`/etc/systemd/system/smart-hospital-mcp.service`

```ini
[Unit]
Description=Smart Hospital Mock MCP Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/kkk/Project/smart_hospital_agent_v2/backend
Environment=PYTHONPATH=/home/kkk/Project/smart_hospital_agent_v2/backend
EnvironmentFile=/home/kkk/Project/smart_hospital_agent_v2/.env
ExecStart=/home/kkk/Project/smart_hospital_agent_v2/backend/.venv/bin/uvicorn mcp_mock.server:app --host 127.0.0.1 --port 9100
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### 4.3 反向 API 隧道

`/etc/systemd/system/smart-hospital-tunnel.service`

```ini
[Unit]
Description=Reverse tunnel hospital API to 106
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -R 18087:127.0.0.1:8020 tx106-friend
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 4.4 WinPC Ollama 隧道（可选）

`/etc/systemd/system/smart-hospital-ollama-tunnel.service`

```ini
[Unit]
Description=Tunnel WinPC Ollama
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 11434:127.0.0.1:11434 winpc
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

启用：

```bash
systemctl daemon-reload
systemctl enable --now smart-hospital-api smart-hospital-mcp smart-hospital-tunnel smart-hospital-ollama-tunnel
systemctl is-active smart-hospital-api smart-hospital-mcp smart-hospital-tunnel smart-hospital-ollama-tunnel
```

## 5. 106 nginx

站点文件：`/etc/nginx/sites-available/hospital.kk.zhzepei.cn`
（`sites-enabled` 软链；改前备份在 `/root/nginx-sites-*.bak-*` 与 `/root/hospital.kk.zhzepei.cn.bak-*`）

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name hospital.kk.zhzepei.cn;
    location /.well-known/acme-challenge/ { root /var/www/certbot; }
    location / { return 301 https://$host$request_uri; }
}

server {
    listen 443 ssl;
    listen [::]:443 ssl;
    server_name hospital.kk.zhzepei.cn;

    root /var/www/hospital;
    index index.html;
    client_max_body_size 10m;

    ssl_certificate /etc/letsencrypt/live/hospital.kk.zhzepei.cn/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/hospital.kk.zhzepei.cn/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    location /api/ {
        proxy_pass http://127.0.0.1:18087;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }

    location / {
        try_files $uri $uri/ $uri.html /index.html;
    }
}
```

`proxy_buffering off` 是 SSE 必需；`client_max_body_size 10m` 用于报告图片上传。

## 6. 证书

```bash
ssh tx106-friend
certbot certonly --nginx -d hospital.kk.zhzepei.cn --non-interactive --agree-tos
nginx -t && systemctl reload nginx
```

- 证书：`/etc/letsencrypt/live/hospital.kk.zhzepei.cn/`
- 有效期：90 天，certbot timer 自动续期
- HTTP 80 仅保留 ACME challenge 与 301

## 7. 生产环境变量

`.env`（家服务器，600，gitignore）关键项：

```ini
AUTH_ENABLED=true
JWT_SECRET=<random>
JWT_TTL_MINUTES=720
ADMIN_USER=admin
ADMIN_PASSWORD=<random>
PATIENT_USER=patient
PATIENT_PASSWORD=<random>
HITL_CONFIRM_TOKEN=<random>    # 服务端审核批准时使用，不下发前端
DATABASE_URL=postgresql://sha:...@127.0.0.1:5434/sha
HOSPITAL_MCP_URL=http://127.0.0.1:9100/mcp
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001
RATE_LIMIT_PER_MINUTE=60
CHAT_DAILY_LIMIT=60            # 0 = 关闭每日问诊配额
MAX_UPLOAD_MB=8
UPLOADS_DIR=/home/kkk/Project/smart_hospital_agent_v2/backend/uploads
```

生产凭据只存服务器 `/root/.credentials/smart-hospital.env`（600），不提交仓库。**`.env` 是家服务器真实运行时配置（600，gitignore，不是占位符）**；仓库里的 `.env.example` 才是占位符模板。API 的 pydantic-settings 会从仓库根目录读取 `.env`，即使 systemd 单元没有 `EnvironmentFile` 也会生效；可选在 API 单元里显式加 `EnvironmentFile=-/home/kkk/Project/smart_hospital_agent_v2/.env`（`-` 表示缺失也不阻止启动），但两处值必须一致。`HITL_CONFIRM_TOKEN` 不设置或保留 `change-me` 时，管理员批准支付会因 MCP token 不匹配而失败（不会静默支付），部署必须设置并与 MCP 侧一致。**v3.1 部署本轮不轮换生产密码**；轮换需主人批准，并执行 `cd backend && .venv/bin/python -m scripts.set_password <username>` 更新 `users` 表哈希。

## 8. 验证

```bash
# 静态
curl -I https://hospital.kk.zhzepei.cn/

# 轻量健康检查（不依赖 DB/模型/MCP）
curl -fsS https://hospital.kk.zhzepei.cn/api/health

# API（公开）
curl -s https://hospital.kk.zhzepei.cn/api/v1/rag/status

# 登录 + 家庭
TOKEN=$(curl -s -X POST https://hospital.kk.zhzepei.cn/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"patient","password":"<从凭据文件读取>"}' | jq -r .access_token)
curl -s -H "Authorization: Bearer $TOKEN" https://hospital.kk.zhzepei.cn/api/v1/family

# v3.1 管理接口：无 JWT → 401；患者 JWT → 403（不得是 nginx Basic 401）
curl -s -o /dev/null -w '%{http_code}\n' https://hospital.kk.zhzepei.cn/api/v1/admin/users
curl -s -o /dev/null -w '%{http_code}\n' \
  -H "Authorization: Bearer $TOKEN" https://hospital.kk.zhzepei.cn/api/v1/admin/users

# v3.1 报告图片必须带 Bearer（不能匿名）
curl -s -o /dev/null -w '%{http_code}\n' https://hospital.kk.zhzepei.cn/api/v1/reports/rep_x/image
```

线上 v3.0 基线（2026-11-08）已用 Playwright 验证：登录、三步问诊、证据、挂号支付、报告视觉解读、药管家、健康时间线、后台、移动端。**v3.1 的 Playwright 3 条流程当前只在本地 `sha_e2e` 通过，部署后需在线上复跑**（见 `docs/V3.1-RELEASE.md`）。

## 8.1 健康检查与备份

- `GET /api/health` 只返回进程存活、版本与 UTC 时间，不访问 DB/模型/MCP，供 Kuma/负载均衡探活。
- `scripts/backup.sh`：`pg_dump`（custom 格式）+ `backend/uploads` tar，产物在 `/home/data/backups/smart-hospital/`，权限 600，保留 7 天。
- systemd：`smart-hospital-backup.timer` 每天 03:30 触发 `smart-hospital-backup.service`；手工执行 `systemctl start smart-hospital-backup.service`。
- 恢复步骤见 `scripts/restore.md`；恢复前先停 API，避免写入冲突。
- Kuma 监控项：`https://hospital.kk.zhzepei.cn/api/health`，期望 200 且响应体 `{"ok": true, ...}`。

## 9. 回滚（v3.1 → v3.0）

v3.1 的库变更只增不删，理论上旧代码可继续运行；**但正式回滚以 DB dump 为准**。

```bash
# 1. 停服务，防止写入
systemctl stop smart-hospital-api.service smart-hospital-mcp.service smart-hospital-tunnel.service

# 2. 代码回到 v3.0 基线
cd /home/kkk/Project/smart_hospital_agent_v2
git checkout v3.0.0        # 若 tag 存在
# 本仓库当前未打 v3.0 tag；基线为 main / commit 4ecb285
# git checkout 4ecb285

# 3. 恢复部署前备份（自定义格式，先确认备份可读）
docker exec -i sha_v2_db sh -c \
  'pg_restore --clean --if-exists --no-owner --no-acl -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  < /home/data/backups/smart-hospital/db-<备份时间戳>.dump

# 4. 恢复上传文件（可选；若 v3.1 期间无新报告可不做）
tar -xzf /home/data/backups/smart-hospital/uploads-<备份时间戳>.tar.gz \
  -C /home/kkk/Project/smart_hospital_agent_v2

# 5. 静态前端回到上一版备份
ssh tx106-friend 'rm -rf /var/www/hospital && cp -a /var/www/hospital.bak-<时间戳> /var/www/hospital'

# 6. 重启并验证
systemctl start smart-hospital-mcp.service smart-hospital-api.service smart-hospital-tunnel.service
curl -fsS http://127.0.0.1:8020/api/health
```

- nginx：`/root/hospital.kk.zhzepei.cn.bak-*`、`/root/nginx-sites-available.bak-*`、`/root/nginx-sites-enabled.bak-*`；回滚后 `nginx -t && systemctl reload nginx`。
- 证书：`certbot certificates`；续期失败不影响已签发证书至到期。
- 回滚后检查 `journalctl -u smart-hospital-api -n 100` 与 `https://hospital.kk.zhzepei.cn/api/health`（v3.0 没有该路由时 404 属预期）。

## 10. 安全边界

- 线上必须 `AUTH_ENABLED=true`；密码存 `users` 表 scrypt 哈希，初始密码只放服务器 600 凭据文件。v3.1 部署本轮不轮换生产密码（待主人批准）。
- 后台：`/admin` 静态页 Basic Auth（nginx，仅页面）；`/api/v1/admin/*` JWT + `role=admin`。**不要在 API 路径加 Basic Auth**（Bearer 冲突）。可选 Cloudflare Access / IP allowlist。
- 医院 MCP 仍是 mock；真实接入（v3.2）前确认鉴权、幂等、对账、超时、患者号映射。
- 报告图片存 `backend/uploads/`（gitignore），只能经 `GET /api/v1/reports/{id}/image` 带 Bearer 读取；不随仓库/静态站发布。
- 配额：`CHAT_DAILY_LIMIT` 按用户自然日生效；限流按真实 IP。
- 106 是朋友的机器：本次只新增一个 nginx 站点与 `/var/www/hospital`，未改其他站点；改动均有备份。v3.1 示例配置未安装，安装属独立审批步骤。
