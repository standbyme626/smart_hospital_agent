#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-/home/data/backups/smart-hospital}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
PG_CONTAINER="${PG_CONTAINER:-sha_v2_db}"

umask 077
mkdir -p "$BACKUP_DIR"

if ! docker inspect "$PG_CONTAINER" >/dev/null 2>&1; then
  echo "postgres container not found: $PG_CONTAINER" >&2
  exit 1
fi

if ! docker exec "$PG_CONTAINER" sh -c '[ -n "$POSTGRES_USER" ] && [ -n "$POSTGRES_DB" ]'; then
  echo "POSTGRES_USER/POSTGRES_DB missing in container: $PG_CONTAINER" >&2
  exit 1
fi

stamp="$(date +%Y%m%d-%H%M%S)"
dump="$BACKUP_DIR/db-$stamp.dump"
uploads="$BACKUP_DIR/uploads-$stamp.tar.gz"
tmp_dump="$dump.part"
trap 'rm -f "$tmp_dump"' EXIT

docker exec "$PG_CONTAINER" sh -c \
  'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --format=custom --no-owner --no-acl' \
  > "$tmp_dump"
chmod 600 "$tmp_dump"
mv "$tmp_dump" "$dump"

tar -C "$ROOT" -czf "$uploads" backend/uploads
chmod 600 "$uploads"

find "$BACKUP_DIR" -maxdepth 1 -type f \
  \( -name 'db-*.dump' -o -name 'uploads-*.tar.gz' \) \
  -mtime +"$RETENTION_DAYS" -delete

echo "backup ok: $dump $uploads"
