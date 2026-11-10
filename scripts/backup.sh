#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-/home/data/backups/smart-hospital}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
PG_CONTAINER="${PG_CONTAINER:-sha_v2_db}"
BACKUP_ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY:-}"
BACKUP_OFFSITE="${BACKUP_OFFSITE:-}"

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
tmp_uploads="$uploads.part"
trap 'rm -f "$tmp_dump" "$tmp_uploads"' EXIT

docker exec "$PG_CONTAINER" sh -c \
  'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --format=custom --no-owner --no-acl' \
  > "$tmp_dump"
chmod 600 "$tmp_dump"
mv "$tmp_dump" "$dump"

tar -C "$ROOT" -czf "$tmp_uploads" backend/uploads
chmod 600 "$tmp_uploads"
mv "$tmp_uploads" "$uploads"

encrypt_file() {
  local src="$1"
  if [ -z "$BACKUP_ENCRYPTION_KEY" ]; then
    return 0
  fi
  if command -v gpg >/dev/null 2>&1; then
    printf '%s' "$BACKUP_ENCRYPTION_KEY" |
      gpg --batch --yes --pinentry-mode loopback --passphrase-fd 0 \
        --symmetric --cipher-algo AES256 -o "$src.gpg" "$src"
    rm -f "$src"
    return 0
  fi
  if command -v age >/dev/null 2>&1 && [[ "$BACKUP_ENCRYPTION_KEY" == age1* ]]; then
    age -r "$BACKUP_ENCRYPTION_KEY" -o "$src.age" "$src"
    rm -f "$src"
    return 0
  fi
  echo "BACKUP_ENCRYPTION_KEY set but no usable gpg or age recipient found" >&2
  return 1
}

encrypt_file "$dump"
[ -f "$dump.gpg" ] && dump="$dump.gpg"
[ -f "$dump.age" ] && dump="$dump.age"
encrypt_file "$uploads"
[ -f "$uploads.gpg" ] && uploads="$uploads.gpg"
[ -f "$uploads.age" ] && uploads="$uploads.age"

if [ -n "$BACKUP_OFFSITE" ]; then
  rsync -az --partial "$dump" "$uploads" "$BACKUP_OFFSITE"/
fi

find "$BACKUP_DIR" -maxdepth 1 -type f \
  \( -name 'db-*.dump*' -o -name 'uploads-*.tar.gz*' \) \
  -mtime +"$RETENTION_DAYS" -delete

echo "backup ok: $dump $uploads"
