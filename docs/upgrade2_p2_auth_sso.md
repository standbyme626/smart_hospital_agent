# Upgrade2 P2-1 Unified Auth Path

## Scope
- Minimal RBAC (token + role dependency) is enforced first.
- Auth decisions are auditable via JSONL logs.
- SSO mapping contract is standardized for both `frontend_new` and Open WebUI.

## Runtime Controls
- `AUTH_RBAC_ENABLED=true|false` (default `false`, so login gate is off by default)
- `AUTH_RBAC_STATIC_TOKENS=token:role,...`
- `AUTH_RBAC_AUDIT_LOG_PATH=logs/security/rbac_audit.jsonl`
- `AUTH_SSO_*` claim and audience mapping keys

## API Contracts
- `POST /api/v1/auth/login`
  - Mock login for dev role tokens.
- `GET /api/v1/auth/whoami`
  - Returns resolved actor and RBAC status.
- `GET /api/v1/auth/sso/mapping`
  - Returns issuer/claim/audience/role map for frontend + Open WebUI integration.
- `POST /api/v1/auth/sso/exchange`
  - Maps upstream SSO claims to internal `{user_id, role, audience}`.

## Protected Endpoints
- `POST /api/v1/admin/rules/interactions` -> `admin`
- `/api/v1/evolution/start|stop|human_judge` -> `admin`
- `/api/v1/evolution/stream` -> `operator` (or higher)

## Audit Evidence
- Every RBAC decision is appended to `AUTH_RBAC_AUDIT_LOG_PATH` with:
  - `request_id`, `path`, `actor_id`, `actor_role`, `allowed`, `reason`.
