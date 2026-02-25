# Stream Schema Alignment Draft (chat + doctor/workflow)

## Goal
统一 `/api/v1/chat/stream` 与 `/api/v1/doctor/workflow` 的流式消息结构，降低前端协议分支成本，同时保留旧协议兼容。

## Unified Envelope (v1)
```json
{
  "version": "v1",
  "type": "thought|token|tool_call|tool_output|phase|final|error",
  "content": "...",
  "node": "diagnosis_node",
  "session_id": "demo-session",
  "ts": 1700000000.123,
  "meta": {}
}
```

## Compatibility Strategy
1. 默认 `legacy`：保持历史 payload，避免直接破坏线上/前端。
2. `ENABLE_UNIFIED_STREAM_SCHEMA=true` 或 `doctor/workflow?schema_mode=unified` 时输出统一 envelope。
3. 前端可逐步切到统一解析器：优先读 `version/type/content/meta`，兼容旧字段。

## Entry Points
- `backend/app/core/stream_schema.py`
- `backend/app/api/v1/endpoints/chat.py`
- `backend/app/api/v1/endpoints/doctor.py`

## Minimal Verification
```bash
# legacy
curl -N -X POST 'http://127.0.0.1:8001/api/v1/doctor/workflow' \
  -H 'Content-Type: application/json' \
  -d '{"message":"我头痛两天了","session_id":"schema-demo-1"}'

# unified
enable_unified=1
curl -N -X POST 'http://127.0.0.1:8001/api/v1/doctor/workflow?schema_mode=unified' \
  -H 'Content-Type: application/json' \
  -d '{"message":"我头痛两天了","session_id":"schema-demo-2"}'
```
