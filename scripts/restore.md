# 备份与恢复

备份由 `scripts/backup.sh` 生成，systemd timer `smart-hospital-backup.timer` 每天 03:30 执行一次。

- 目录：`/home/data/backups/smart-hospital/`（目录 700，文件 600）
- 内容：`db-<时间戳>.dump`（PostgreSQL 自定义格式）+ `uploads-<时间戳>.tar.gz`
- 保留：7 天（`RETENTION_DAYS` 可覆盖）
- 数据库：通过 `docker exec` 进入运行中的 Postgres 容器（默认 `sha_v2_db`，`PG_CONTAINER` 可覆盖），使用容器内 `POSTGRES_USER`/`POSTGRES_DB`；宿主机命令行不携带数据库密码。

## 手工备份

```bash
sudo systemctl start smart-hospital-backup.service
sudo journalctl -u smart-hospital-backup.service -n 20 --no-pager
```

验证 dump 可读：

```bash
PG_CONTAINER="${PG_CONTAINER:-sha_v2_db}"
docker exec -i "$PG_CONTAINER" sh -c 'pg_restore -l' \
  < /home/data/backups/smart-hospital/db-<时间戳>.dump | head
gzip -t /home/data/backups/smart-hospital/uploads-<时间戳>.tar.gz
```

## 恢复数据库

```bash
PG_CONTAINER="${PG_CONTAINER:-sha_v2_db}"

# 1. 先停写入（恢复期间不要有 API 写入）
systemctl stop smart-hospital-api.service

# 2. 恢复（--clean 会删除现有对象，先在测试库验证过再用）
docker exec -i "$PG_CONTAINER" sh -c \
  'pg_restore --clean --if-exists --no-owner --no-acl -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  < /home/data/backups/smart-hospital/db-<时间戳>.dump

# 3. 启动并验证
systemctl start smart-hospital-api.service
curl -fsS http://127.0.0.1:8020/api/health
```

恢复到新数据库时，先在容器内创建空库，再用 `TARGET_DB` 指定目标库（不要用 `--clean`）：

```bash
docker exec -i -e TARGET_DB=<新库名> "$PG_CONTAINER" sh -c \
  'pg_restore --no-owner --no-acl -U "$POSTGRES_USER" -d "$TARGET_DB"' \
  < /home/data/backups/smart-hospital/db-<时间戳>.dump
```

## 恢复上传文件

```bash
cd /home/kkk/Project/smart_hospital_agent_v2
tar -xzf /home/data/backups/smart-hospital/uploads-<时间戳>.tar.gz -C "$PWD"
```

恢复后确认 `backend/uploads/` 权限与属主，并抽查报告图片可访问。

## 注意

- 备份包含患者数据，禁止复制到公网或不安全介质。
- `scripts/backup.sh` 不读取宿主机 `.env`，也不在命令行参数中传数据库密码。
