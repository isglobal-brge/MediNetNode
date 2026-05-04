# MediNetNode — Pending Tasks

> Last updated: 2026-05-02

---

## High Priority (pre-launch / paper review week)

- [x] **API key expiry (30-day rotation)**
  - `users/models.py` — `expires_at` field present; `is_expired()` method implemented
  - Middleware `APIAuthenticationMiddleware` rejects expired keys with `401 API key has expired`
  - `users/views.py` — key creation now sets `expires_at = now + 30 days` automatically
  - Success page context returns `expires_at` ISO timestamp to the Administrator
  - Deferred: pre-expiry warning via `X-API-Key-Expires-In` response header (days remaining) — avoids email/Celery dependency for open-source installs

---

## Differential Privacy — UI & Budget Visibility (sprint this week)

- [x] **GET /api/v2/budget-status/ — per-dataset budget for the authenticated researcher**
  - Returns `{dataset_name, spent_epsilon, remaining_budget, lifetime_budget, max_per_job}` per dataset the researcher has access to
  - Hub calls this when datasets are selected and on manual refresh

- [x] **POST /api/v2/estimate-epsilon/ — pre-flight ε estimate for a given config**
  - Input: `{model_json, dataset_name}` (same shape as start-client)
  - Output: `{estimated_epsilon, delta}` — no training started
  - Reuses existing `estimate_job_epsilon()` logic from `api/views.py`
  - Hub calls this live as researcher changes rounds / batch / noise multiplier

- [x] **Hub training page — budget panel (ruling budget + per-node expandable)**
  - Show the minimum ε budget across all selected nodes (the "ruling budget")
  - Expandable per-node detail: connection name, dataset, spent / remaining / limit
  - Refresh button calls `/api/v2/budget-status/` on each node
  - Only show production budget (experimental datasets have no budget tracking)

- [x] **Hub training page — interactive ε calculator**
  - As researcher configures rounds / batch / noise multiplier, show in real time:
    - Estimated ε this job will consume (via `/api/v2/estimate-epsilon/`)
    - Remaining budget after this job
    - Color-coded warning: green (safe) / amber (>80 % used) / red (would exceed)
  - Calculation debounced 600 ms after last config change
  - Only active when a production dataset is selected (not experimental)

---

## Differential Privacy Accounting (deferred — post paper)

- [ ] **Per-round ε tracking**
  - Record ε consumed at each Flower round, not only at job completion
  - Requires changes in `api/federated/dl_client.py` (round callbacks) and a new `RoundPrivacySpend` model

- [ ] **Per-client DP accounting**
  - Currently ε is tracked per researcher × dataset; extend to per (researcher, dataset, client_node) triplet
  - Needed for multi-site studies where each hospital contributes differently

- [ ] **Automatic sensitivity Δf validation**
  - When a researcher uploads a model config, validate that the claimed sensitivity matches the dataset's `SENSITIVITY_DEFAULTS`
  - Should return a warning (or block) if the researcher's noise multiplier is below the minimum for the declared sensitivity level

- [ ] **Heterogeneous budgets across nodes (multi-node training)**
  - Current rule: Hub uses the **minimum** ε budget across all selected nodes as the ruling budget, guaranteeing no node is exceeded
  - Future: nodes with an exhausted budget are **automatically deactivated** at job start; training continues with the remaining nodes (Flower supports `min_available_clients` < total)
  - Requires: pre-flight check per node, dynamic client list, partial-activation logic in `activate_clients_for_training`

---

## UI / UX — Budget Reset Workflow

- [x] **Budget reset request — Researcher side (Hub)**
  - Training page: "Request Budget Reset" dropdown item on failed jobs with `budget_exhausted_nodes`
  - Modal with justification textarea → proxied via Hub to Node `/api/v2/budget-reset/`
  - Implemented: `request_budget_reset_proxy` view + `budget_exhausted_nodes` field on `TrainingJob`

- [x] **Budget reset request — Admin side (Node)**
  - List view: `/trainings/budget-reset/` with status filter + pending badge
  - Detail view: researcher info, ε budget status bar, training history on that dataset, Approve/Reject form
  - Implemented: `budget_reset_list` + `budget_reset_detail` views + templates

---

## API Versioning

- [x] **Migrate all endpoints from `/api/v1/` → `/api/v2/`**
  - URL patterns in `api/urls.py`, middleware, views, tests, templates, Hub callers
  - No backward compatibility needed (no external consumers yet)

---

## Infrastructure / DevOps

- [ ] **Celery + Redis for async tasks**
  - Budget expiry notifications
  - Long-running audit log exports
  - Currently stubbed — `prod.py` references Redis but workers not wired

- [ ] **Automated secret rotation**
  - `entrypoint.sh` generates Fernet keys at first run; add rotation procedure for production deployments

---

## Testing Gaps

- [ ] Add integration tests for `/api/v2/budget-reset/` endpoints
- [ ] Add integration tests for `/api/v2/budget-status/` and `/api/v2/estimate-epsilon/`
- [ ] Add tests for expired API key rejection
- [ ] Add tests for `estimate_job_epsilon()` with edge-case DP parameters (very low noise multiplier, tiny batch size)
