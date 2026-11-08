# 106 部署记录（hospital.kk.zhzepei.cn）

> 2026-11-08 上线。静态前端 + 家服务器 API + 反向 SSH 隧道 + Let's Encrypt。
> 本文不含任何密码/密钥；凭据在服务器 `/root/.credentials/smart-hospital.env`（600）。

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

## 2. 构建与发布静态前端

```bash
cd /home/kkk/Project/smart_hospital_agent_v2/frontend
NEXT_PUBLIC_API_BASE= npm run build
ssh tx106-friend "mkdir -p /var/www/hospital"
scp -r out/* tx106-friend:/var/www/hospital/
```

`NEXT_PUBLIC_API_BASE=` 留空表示同源 `/api`，由 nginx 反代。开发环境用 `frontend/.env.local` 指向 `http://127.0.0.1:8020`。

## 3. 家服务器 systemd

### 3.1 API

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
ExecStart=/home/kkk/Project/smart_hospital_agent_v2/backend/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8020
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### 3.2 Mock 医院 MCP

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
ExecStart=/home/kkk/Project/smart_hospital_agent_v2/backend/.venv/bin/uvicorn mcp_mock.server:app --host 127.0.0.1 --port 9100
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### 3.3 反向 API 隧道

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

### 3.4 WinPC Ollama 隧道（可选）

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

## 4. 106 nginx

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

## 5. 证书

```bash
ssh tx106-friend
certbot certonly --nginx -d hospital.kk.zhzepei.cn --non-interactive --agree-tos
nginx -t && systemctl reload nginx
```

- 证书：`/etc/letsencrypt/live/hospital.kk.zhzepei.cn/`
- 有效期：90 天，certbot timer 自动续期
- HTTP 80 仅保留 ACME challenge 与 301

## 6. 生产环境变量

`.env`（家服务器，600，gitignore）关键项：

```ini
AUTH_ENABLED=true
JWT_SECRET=<random>
ADMIN_USER=admin
ADMIN_PASSWORD=<random>
PATIENT_USER=patient
PATIENT_PASSWORD=<random>
GUEST_LOGIN_ENABLED=false
HITL_CONFIRM_TOKEN=confirm-demo   # 演示支付；真实医院 MCP 会签发确认
DATABASE_URL=postgresql://sha:...@127.0.0.1:5434/sha
HOSPITAL_MCP_URL=http://127.0.0.1:9100/mcp
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001
```

生产凭据只存服务器 `/root/.credentials/smart-hospital.env`（600），不提交仓库。

## 7. 验证

```bash
# 静态
curl -I https://hospital.kk.zhzepei.cn/

# API（公开）
curl -s https://hospital.kk.zhzepei.cn/api/v1/rag/status

# 登录 + 家庭
TOKEN=$(curl -s -X POST https://hospital.kk.zhzepei.cn/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"patient","password":"<从凭据文件读取>"}' | jq -r .access_token)
curl -s -H "Authorization: Bearer $TOKEN" https://hospital.kk.zhzepei.cn/api/v1/family
```

线上 Playwright 已完整验证：登录、三步问诊、证据、挂号支付、报告视觉解读、药管家、健康时间线、后台、移动端。

## 8. 回滚

- 静态：保留上一版 `out/` 或从 git 重新构建；旧文件在 `/var/www/hospital`，可直接覆盖。
- nginx：`/root/hospital.kk.zhzepei.cn.bak-*`、`/root/nginx-sites-available.bak-*`、`/root/nginx-sites-enabled.bak-*`。
- 服务：`systemctl stop smart-hospital-*`；旧 API 可用 `uvicorn` 手动拉起。
- 数据库：`sha_v2_pgdata` 卷；变更前先 `pg_dump`。
- 证书：`certbot certificates`；续期失败不影响已签发证书至到期。

## 9. 安全边界

- 线上必须 `AUTH_ENABLED=true`；默认密码已替换为强随机。
- 医院 MCP 仍是 mock；真实接入前确认鉴权、幂等、对账、超时。
- 报告图片存 `backend/uploads/`（gitignore），不随仓库/静态站发布。
- 106 是朋友的机器：本次只新增一个 nginx 站点与 `/var/www/hospital`，未改其他站点；改动均有备份。
