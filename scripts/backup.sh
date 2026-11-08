#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
BACKUP_DIR="${BACKUP_DIR:-/home/data/backups/smart-hospital}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"

umask 077
mkdir -p "$BACKUP_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing env file: $ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "DATABASE_URL is not set in $ENV_FILE" >&2
  exit 1
fi

stamp="$(date +%Y%m%d-%H%M%S)"
dump="$BACKUP_DIR/db-$stamp.dump"
uploads="$BACKUP_DIR/uploads-$stamp.tar.gz"

pg_dump --format=custom --no-owner --no-acl --file="$dump" "$DATABASE_URL"
tar -C "$ROOT" -czf "$uploads" backend/uploads

chmod 600 "$dump" "$uploads"
find "$BACKUP_DIR" -maxdepth 1 -type f \
  \( -name 'db-*.dump' -o -name 'uploads-*.tar.gz' \) \
  -mtime +"$RETENTION_DAYS" -delete

echo "backup ok: $dump $uploads"
