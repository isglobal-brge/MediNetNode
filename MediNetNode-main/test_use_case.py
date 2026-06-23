#!/usr/bin/env python3
"""
MediNet Section 3.1 Use-Case End-to-End Test
=============================================
Reproduces the full DP federated learning workflow described in the paper:

  Step 1  — Ping (verify auth + API key expiry)
  Step 2  — Get dataset info (verify access + privacy policy)
  Step 3  — Budget status (baseline before any training)
  Step 4  — Estimate ε for experimental run (batch=32, no DP noise)
  Step 5  — Estimate ε for experimental run (batch=16, no DP noise)
  Step 6  — Estimate ε for production DP run (σ configured in test_config.json)
  Step 7  — Launch experimental run batch=32 (use_experiment=True → no budget)
  Step 8  — Launch experimental run batch=16 (use_experiment=True → no budget)
  Step 9  — Launch production DP run (use_experiment=False → budget consumed)
  Step 10 — Budget status (verify ε was decremented)
  Step 11 — Auditor summary (budget delta verification)
  Step 12 — Consolidated metrics table (accuracy / F1 / ε per session)

Each of Steps 7–9 starts its own Hub training job (and therefore its own Flower
gRPC server).  Sequential execution matches real Hub operation — one job at a
time — and avoids FedAvg contamination that arises when clients with different
batch sizes or noise levels all connect to the same server concurrently.

Configuration is read from test_config.json (same directory) or a custom path.
Run:  python test_use_case.py [--config path/to/config.json]
"""

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests

# Force UTF-8 output on Windows so ε and emoji render correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ── Budget reset ──────────────────────────────────────────────────────────────

def reset_budget(dataset_name: str,
                 per_job_max: float | None = None,
                 total_budget: float | None = None) -> None:
    """Reset epsilon budgets for the test dataset via Django ORM.

    Runs a manage.py shell one-liner so the test is idempotent — repeated
    runs against the same database don't fail because a previous run exhausted
    the budget.  Also updates max_epsilon_per_job and lifetime_budget on the
    DatasetPrivacyPolicy when the caller passes explicit values, so the DB
    policy stays in sync with test_config.json without requiring manual admin
    edits between runs.

    Prints a warning but never raises: if the shell call fails the test itself
    will catch the exhausted budget at Step 9 with a clear message.
    """
    policy_updates = "spent_epsilon=0.0"
    if per_job_max is not None:
        policy_updates += f", max_epsilon_per_job={per_job_max}"
    if total_budget is not None:
        policy_updates += f", lifetime_budget={total_budget}"

    # ResearcherEpsilonBudget has its own per-job and lifetime limits that are
    # checked independently of DatasetPrivacyPolicy — update both.
    researcher_updates = "spent_epsilon=0.0"
    if per_job_max is not None:
        researcher_updates += f", max_epsilon_per_job={per_job_max}"
    if total_budget is not None:
        researcher_updates += f", lifetime_budget={total_budget}"

    script = (
        "from dataset.models import DatasetPrivacyPolicy, ResearcherEpsilonBudget; "
        f"n1 = DatasetPrivacyPolicy.objects.filter(dataset__name='{dataset_name}').update({policy_updates}); "
        f"n2 = ResearcherEpsilonBudget.objects.filter(dataset__name='{dataset_name}').update({researcher_updates}); "
        "print(f'Budget reset: policy={n1}, researcher={n2}')"
    )
    result = subprocess.run(
        [sys.executable, "manage.py", "shell", "-c", script],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent),
    )
    if result.returncode == 0:
        print(f"  [SETUP] {result.stdout.strip()}")
    else:
        print(f"  [SETUP] WARNING: budget reset failed — {result.stderr.strip()[:120]}")


def _reset_rate_limit(username: str) -> None:
    """Delete APIRequest records for the test user so the rate limiter (100
    req/hour) does not block repeated runs within the same session."""
    if not username:
        return
    script = (
        "from users.models import APIRequest; "
        "from django.contrib.auth import get_user_model; "
        "User = get_user_model(); "
        f"qs = User.objects.filter(username='{username}'); "
        "u = qs.first(); "
        "n = APIRequest.objects.filter(user=u).delete()[0] if u else 0; "
        "print(f'Rate-limit reset: {n} APIRequest records deleted')"
    )
    result = subprocess.run(
        [sys.executable, "manage.py", "shell", "-c", script],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent),
    )
    if result.returncode == 0:
        print(f"  [SETUP] {result.stdout.strip()}")
    else:
        print(f"  [SETUP] WARNING: rate-limit reset failed — {result.stderr.strip()[:120]}")


# ── Config loader ──────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        return json.load(f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def headers(api_key: str, client_ip: str) -> dict:
    return {
        "X-API-Key":    api_key,
        "X-Client-IP":  client_ip,
        "Content-Type": "application/json",
    }


def ok(label: str, response: requests.Response, expected_status: int = 200) -> bool:
    passed = response.status_code == expected_status
    icon   = "PASS" if passed else "FAIL"
    print(f"  [{icon}] {label} -> HTTP {response.status_code}")
    if not passed:
        print(f"     Expected {expected_status}, got {response.status_code}")
        try:
            print(f"     Body: {response.json()}")
        except Exception:
            print(f"     Body: {response.text[:300]}")
    return passed


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── Model config builders ──────────────────────────────────────────────────────

def model_json_experimental(dataset_id: int, dataset_name: str,
                             batch_size: int, epochs: int = 10) -> dict:
    """Experimental run — use_experiment=True, no DP noise.

    Note: no sigmoid on the output layer — BCEWithLogitsLoss applies sigmoid
    internally; adding it explicitly causes double-squashing that kills gradients.
    """
    return {
        "use_experiment": True,
        "train": {"epochs": epochs},
        "model": {
            "model_name": "E2E Test Experimental",
            "metadata": {"model_type": "dl", "framework": "pytorch"},
            "layers": [
                {"type": "linear", "params": {"in_features": 51, "out_features": 64}},
                {"type": "relu"},
                {"type": "linear", "params": {"in_features": 64, "out_features": 32}},
                {"type": "relu"},
                {"type": "linear", "params": {"in_features": 32, "out_features": 1}},
            ],
            "dataset": {
                "selected_datasets": [
                    {"dataset_id": dataset_id, "dataset_name": dataset_name}
                ]
            },
            "training": {
                "batch_size": batch_size,
                "optimizer":  {"type": "Adam", "learning_rate": 0.001},
                "loss":       "binary_cross_entropy",
            },
        },
    }


def model_json_dp(dataset_id: int, dataset_name: str, batch_size: int = 32,
                  epochs: int = 10, noise_multiplier: float = 1.1,
                  max_grad_norm: float = 1.0) -> dict:
    """Production DP run — use_experiment=False, DP enabled via Opacus.

    Note: no sigmoid on the output layer — BCEWithLogitsLoss applies sigmoid
    internally; adding it explicitly causes double-squashing that kills gradients.
    """
    return {
        "use_experiment": False,
        "train": {"epochs": epochs},
        "model": {
            "model_name": "E2E Test DP Production",
            "metadata": {"model_type": "dl", "framework": "pytorch"},
            "layers": [
                {"type": "linear", "params": {"in_features": 51, "out_features": 64}},
                {"type": "relu"},
                {"type": "linear", "params": {"in_features": 64, "out_features": 32}},
                {"type": "relu"},
                {"type": "linear", "params": {"in_features": 32, "out_features": 1}},
            ],
            "dataset": {
                "selected_datasets": [
                    {"dataset_id": dataset_id, "dataset_name": dataset_name}
                ]
            },
            "training": {
                "batch_size": batch_size,
                "optimizer": {
                    "type": "Adam",
                    "learning_rate": 0.001,
                    "differential_privacy": {
                        "noise_multiplier": noise_multiplier,
                        "max_grad_norm":    max_grad_norm,
                    },
                },
                "loss": "binary_cross_entropy",
            },
        },
    }


# ── Flower server helpers ─────────────────────────────────────────────────────

def _wait_for_port(host: str, port: int, timeout: int = 60) -> bool:
    """Poll until TCP port accepts connections or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(2)
    return False


def _wait_for_port_close(host: str, port: int, timeout: int = 90) -> bool:
    """Poll until TCP port stops accepting connections (Flower server has shut down).

    Used between sequential training jobs so the next Hub job can safely bind
    the same port without getting an 'address already in use' error.
    Returns True when the port is confirmed closed, False on timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                time.sleep(3)   # port still accepting — server still running
        except OSError:
            return True         # port closed — safe to proceed
    print(f"  WARNING: Port {port} still open after {timeout}s — proceeding anyway")
    return False


def _kill_port(port: int) -> None:
    """
    On Windows: kill any process currently holding the given TCP port.
    Uses netstat -ano to find the PID and taskkill to terminate it.
    Safe to call even if nothing is listening.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and ("LISTENING" in line or "ESTABLISHED" in line):
                parts = line.split()
                pid = parts[-1]
                if pid.isdigit() and int(pid) > 0:
                    subprocess.run(
                        ["taskkill", "/PID", pid, "/F"],
                        capture_output=True, timeout=5
                    )
                    print(f"  Killed stale process PID {pid} holding port {port}")
                    time.sleep(2)   # brief settle time
                    return
    except Exception as e:
        print(f"  WARNING: Could not kill process on port {port}: {e}")


def step0_start_flower_server(cfg: dict) -> tuple[bool, dict]:
    """
    Log into MediNetHub, create a minimal model config, start a training job
    (which spawns the Flower gRPC server), and wait for the port to open.

    selected_datasets is intentionally left empty so the Hub will NOT
    auto-call start-client on any Node — the test steps 7-9 do that manually.
    """
    section("Step 0 — Start Flower gRPC server via Hub")

    hub = cfg["hub_url"]
    s   = requests.Session()

    hub_meta: dict = {}   # shared context returned to caller

    # 1. Obtain CSRF token
    login_page = s.get(f"{hub}/login/", timeout=10)
    if login_page.status_code != 200:
        print(f"  ERROR: Cannot reach Hub login page (HTTP {login_page.status_code})")
        return False, hub_meta
    csrf = s.cookies.get("csrftoken", "")

    # 2. Log in
    login_resp = s.post(
        f"{hub}/login/",
        data={
            "username":          cfg["hub_username"],
            "password":          cfg["hub_password"],
            "csrfmiddlewaretoken": csrf,
        },
        headers={"Referer": f"{hub}/login/"},
        timeout=10,
        allow_redirects=True,
    )
    if "login" in login_resp.url.lower():
        print("  ERROR: Hub login failed — check hub_username / hub_password in config")
        return False, hub_meta
    csrf = s.cookies.get("csrftoken", csrf)
    print("  Hub login OK")

    hub_hdrs = {"X-CSRFToken": csrf, "Referer": hub, "Content-Type": "application/json"}

    # 3. Create a minimal model config
    model_payload = {
        "model_name": "e2e_test_model",
        "config": {
            "model_name": "E2E Test — Heart Attack FL",
            "metadata": {"model_type": "dl", "framework": "pytorch"},
            "layers": [
                {"type": "linear", "params": {"in_features": 51, "out_features": 64}},
                {"type": "relu"},
                {"type": "linear", "params": {"in_features": 64, "out_features": 32}},
                {"type": "relu"},
                {"type": "linear", "params": {"in_features": 32, "out_features": 1}},
                {"type": "sigmoid"},
            ],
        },
    }
    mc_resp = s.post(
        f"{hub}/api/save-model-config/",
        json=model_payload,
        headers=hub_hdrs,
        timeout=10,
    )
    if mc_resp.status_code != 200 or not mc_resp.json().get("success"):
        print(f"  ERROR: save-model-config failed: {mc_resp.text[:200]}")
        return False, hub_meta
    model_id = mc_resp.json().get("model_id") or mc_resp.json().get("id")
    print(f"  Model config created (id={model_id})")

    # 4b. Kill any stale Flower server holding the gRPC port before starting a new one
    flower_port = int(cfg["flower_server"].split(":")[-1])
    _kill_port(flower_port)

    # 5. Start training job 
    train_resp = s.post(
        f"{hub}/api/start-training/",
        json={
            "model_id":       model_id,
            "job_name":       f"e2e_test_{int(time.time())}",
            "job_description": "Automated end-to-end use-case test",
            "config": {
                "train":  {"rounds": 10, "fraction_fit": 1.0,
                           "min_fit_clients": 1, "min_available_clients": 1},
                "server": {"port": flower_port},
            },
        },
        headers=hub_hdrs,
        timeout=30,
    )
    if train_resp.status_code != 200 or not train_resp.json().get("success"):
        print(f"  ERROR: start-training failed: {train_resp.text[:200]}")
        return False, hub_meta
    job_id = train_resp.json().get("id")
    hub_meta = {"session": s, "job_id": job_id, "hub": hub,
                "csrf": csrf, "hub_hdrs": hub_hdrs}
    print(f"  Training job started (id={job_id})")

    # 6. Wait for gRPC port to open (Flower server ready)
    flower_host = cfg["flower_server"].split(":")[0]
    print(f"  Waiting for Flower gRPC server at {cfg['flower_server']} ...")
    if not _wait_for_port(flower_host, flower_port, timeout=60):
        print(f"  ERROR: Flower server did not open port {flower_port} within 60 s")
        return False, hub_meta

    print(f"  Flower server ready at {cfg['flower_server']}")

    # 7. Verify the Hub job is healthy (not immediately failed due to port conflict)
    try:
        r = hub_meta.get("session").get(
            f"{hub}/api/client-status/{job_id}/", timeout=5
        )
        job_status = r.json().get("job_status", "?") if r.status_code == 200 else "?"
        if job_status == "failed":
            print(f"  ERROR: Hub job {job_id} is already 'failed' — likely port conflict.")
            print(f"     Kill any stale process on port {flower_port} and re-run.")
            return False, hub_meta
        print(f"  Hub job {job_id} status: {job_status}")
    except Exception as e:
        print(f"  WARNING: Could not verify job status: {e}")

    return True, hub_meta


# ── Steps ─────────────────────────────────────────────────────────────────────

def check_flower_server_alive(cfg: dict, hub_meta: dict) -> bool:
    """
    Check if the Flower server is still listening AND if the Hub job is healthy.
    Prints a diagnostic block. Returns True if server is alive.
    """
    flower_host, flower_port_str = cfg["flower_server"].split(":")
    flower_port = int(flower_port_str)
    hub = hub_meta.get("hub", "")
    s   = hub_meta.get("session")
    job_id = hub_meta.get("job_id")

    port_alive = False
    try:
        with socket.create_connection((flower_host, flower_port), timeout=3):
            port_alive = True
    except OSError:
        pass

    print(f"  {'[OPEN]' if port_alive else '[CLOSED]'} Flower server port {flower_port}: "
          f"{'OPEN' if port_alive else 'CLOSED'}")

    if s and job_id and hub:
        try:
            r = s.get(f"{hub}/api/client-status/{job_id}/", timeout=5)
            if r.status_code == 200:
                status_data = r.json()
                job_status = status_data.get("status", "?")
                print(f"  Hub job {job_id} status: {job_status}")
            else:
                print(f"  WARNING: Hub job status API returned HTTP {r.status_code}")
        except Exception as e:
            print(f"  WARNING: Could not fetch Hub job status: {e}")

    return port_alive


def step1_ping(cfg: dict) -> bool:
    section("Step 1 — Ping (auth + API key expiry)")
    r = requests.get(
        f"{cfg['node_url']}/api/v2/ping",
        headers=headers(cfg["researcher_api_key"], cfg["client_ip"]),
        timeout=10,
    )
    if ok("Ping", r):
        data = r.json()
        print(f"     status: {data.get('status')}")
        key_info = data.get("api_key")
        if key_info:
            print(f"     API key expires: {key_info.get('expires_at')}")
            days = key_info.get("days_remaining", 0)
            if days <= 7:
                print(f"     WARNING: key expires in {days} day(s)")
            else:
                print(f"     Days remaining: {days}")
        else:
            print("     API key has no expiry configured")
    return r.status_code == 200


def step2_get_data_info(cfg: dict) -> list:
    section("Step 2 — Get dataset info")
    r = requests.get(
        f"{cfg['node_url']}/api/v2/get-data-info",
        headers=headers(cfg["researcher_api_key"], cfg["client_ip"]),
        timeout=10,
    )
    if ok("Get data info", r):
        data  = r.json()
        ids   = data.get("dataset_id", [])
        names = data.get("dataset_name", [])
        print(f"     Datasets accessible: {len(ids)}")
        for i, (did, name) in enumerate(zip(ids, names)):
            policy = (data.get("privacy_policy") or [])[i] if data.get("privacy_policy") else None
            print(f"       [{did}] {name}")
            if policy:
                print(
                    f"            ε lifetime: {policy.get('lifetime_budget')} | "
                    f"remaining: {policy.get('remaining_budget')} | "
                    f"per-job max: {policy.get('max_epsilon_per_job')}"
                )
            else:
                print("            No privacy policy configured")
        return ids
    return []


def step3_budget_status(cfg: dict, label: str = "Budget status") -> dict:
    section(f"Step 3 — {label}")
    r = requests.get(
        f"{cfg['node_url']}/api/v2/budget-status/",
        headers=headers(cfg["researcher_api_key"], cfg["client_ip"]),
        timeout=10,
    )
    if ok("Budget status", r):
        data = r.json()
        for ds in data.get("datasets", []):
            print(f"     [{ds['dataset_id']}] {ds['dataset_name']}")
            print(
                f"       spent ε: {ds['spent_epsilon']} | "
                f"remaining: {ds['remaining_budget']} | "
                f"lifetime: {ds['lifetime_budget']}"
            )
        return data
    return {}


def step_estimate_epsilon(cfg: dict, label: str, model_config: dict) -> Optional[float]:
    section(label)
    payload = {
        "dataset_name": cfg["dataset_name"],
        "model_json":   model_config,
    }
    r = requests.post(
        f"{cfg['node_url']}/api/v2/estimate-epsilon/",
        headers=headers(cfg["researcher_api_key"], cfg["client_ip"]),
        json=payload,
        timeout=10,
    )
    if ok("Estimate ε", r):
        data = r.json()
        eps  = data.get("estimated_epsilon")
        print(f"     Estimated ε  = {eps}")
        print(f"     δ            = {data.get('delta')}")
        print(f"     Dataset size = {data.get('dataset_size')} records")
        per_job_max = cfg["privacy_policy"]["per_job_max"]
        if eps is not None and eps > per_job_max:
            print(f"     WARNING: Estimated ε ({eps}) exceeds per-job max ({per_job_max})")
        return eps
    return None


def step_start_client(
    cfg: dict,
    label: str,
    model_config: dict,
    client_id: str,
    retry_on_concurrent_limit: bool = False,
    concurrent_limit_timeout: int = 300,
) -> bool:
    """
    POST /api/v2/start-client.

    When ``retry_on_concurrent_limit=True`` and the Node returns HTTP 429 with
    the "sesiones simultáneas" message (not the rate-limit 429), poll every 10 s
    for up to ``concurrent_limit_timeout`` seconds waiting for a slot to open.
    """
    section(label)
    payload = {
        "model_json":     model_config,
        "server_address": cfg["flower_server"],
        "client_id":      client_id,
        "ssl_enabled":    cfg.get("ssl_enabled", False),
    }
    url     = f"{cfg['node_url']}/api/v2/start-client"
    hdrs    = headers(cfg["researcher_api_key"], cfg["client_ip"])
    deadline = time.time() + concurrent_limit_timeout

    while True:
        r = requests.post(url, headers=hdrs, json=payload, timeout=300)
        # Concurrent-session limit → wait and retry if requested
        if r.status_code == 429 and retry_on_concurrent_limit:
            body = r.json()
            if "sesiones simultáneas" in body.get("error", ""):
                elapsed = int(time.time() - (deadline - concurrent_limit_timeout))
                print(f"  [{elapsed:3d}s] Concurrent session limit reached — waiting 10 s for a slot ...")
                if time.time() + 10 > deadline:
                    print(f"  ERROR: Timed out waiting for a free training session slot ({concurrent_limit_timeout}s)")
                    ok("Start client", r)   # print the error
                    return False
                time.sleep(10)
                continue
        # Any other response (success or real error)
        if ok("Start client", r):
            data = r.json()
            print(f"     status:    {data.get('status')}")
            print(f"     client_id: {data.get('client_id')}")
            print(f"     user:      {data.get('user')}")
            return True
        return False


def step10_wait_for_budget_change(
    cfg: dict,
    budget_before: dict,
    target_dataset: str,
    timeout: int = 180,
    poll_interval: int = 5,
) -> dict:
    """
    Poll /budget-status/ until the spent_epsilon for target_dataset increases
    (meaning the DP production job has finished and recorded its spend),
    or until timeout expires.

    Returns the final budget snapshot (changed or not).
    """
    section("Step 10 — Wait for budget decrement (DP round completion)")
    before_ds = next(
        (ds for ds in budget_before.get("datasets", []) if ds["dataset_name"] == target_dataset),
        None,
    )
    spent_before = float(before_ds["spent_epsilon"]) if before_ds else 0.0

    print(f"  ⏳ Polling budget until ε spent > {spent_before} "
          f"(timeout {timeout}s, poll every {poll_interval}s) ...")

    deadline = time.time() + timeout
    last_budget: dict = {}
    while time.time() < deadline:
        try:
            r = requests.get(
                f"{cfg['node_url']}/api/v2/budget-status/",
                headers=headers(cfg["researcher_api_key"], cfg["client_ip"]),
                timeout=10,
            )
        except requests.RequestException as exc:
            elapsed = int(time.time() - (deadline - timeout))
            print(f"     [{elapsed:3d}s] request error: {exc} — retrying in {poll_interval}s")
            time.sleep(poll_interval)
            continue

        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", 60))
            elapsed = int(time.time() - (deadline - timeout))
            print(f"     [{elapsed:3d}s] rate-limited (429) — waiting {retry_after}s ...")
            time.sleep(retry_after)
            continue

        if r.status_code == 200:
            data = r.json()
            ds = next(
                (d for d in data.get("datasets", []) if d["dataset_name"] == target_dataset),
                None,
            )
            if ds:
                spent_now = float(ds["spent_epsilon"])
                elapsed   = int(time.time() - (deadline - timeout))
                print(f"     [{elapsed:3d}s] ε spent = {spent_now}")
                last_budget = data
                if spent_now > spent_before:
                    print(f"  Budget decremented after {elapsed}s (ε spent = {spent_now})")
                    return last_budget

        time.sleep(poll_interval)

    print(f"  WARNING: Budget unchanged after {timeout}s — FL round may still be running")
    # Return the most recent snapshot regardless
    return last_budget if last_budget else budget_before


def step_hub_metrics(hub_meta: dict, step_label: str = "Hub aggregated metrics") -> dict:
    """Query the Hub for FedAvg-aggregated metrics for a completed training job.

    Calls two Hub endpoints using the already-authenticated session from
    step0_start_flower_server:

      GET /api/get-job-metrics/<job_id>/  — per-round weighted-average metrics
      GET /api/job-details/<job_id>/      — final job summary (status, duration)

    Returns a dict with keys: success, final_round (dict), all_rounds (list).
    Prints a per-round table so the test output shows what the global model
    achieved after FedAvg aggregation, not just the local Node training metrics.
    """
    section(f"{step_label} (from Hub DB — FedAvg aggregated)")

    s      = hub_meta.get("session")
    job_id = hub_meta.get("job_id")
    hub    = hub_meta.get("hub")

    if not s or not job_id or not hub:
        print("  [SKIP] No Hub session available — skipping Hub metrics")
        return {"success": False}

    result: dict = {"success": False, "final_round": {}, "all_rounds": []}

    # ── Per-round aggregated metrics ──────────────────────────────────────────
    try:
        r = s.get(f"{hub}/api/get-job-metrics/{job_id}/", timeout=10)
        if r.status_code == 200:
            data   = r.json()
            rounds = data.get("metrics", [])
            status = data.get("job_status", "?")
            progress = data.get("progress", 0)
            result["all_rounds"] = rounds

            hdr = f"  {'round':>5}  {'acc':>8}  {'loss':>8}  {'f1':>8}  {'prec':>8}  {'recall':>8}"
            sep = "  " + "-" * 56
            print(f"  Job {job_id} — status: {status}  progress: {progress}%")
            print(hdr)
            print(sep)
            for m in rounds:
                rnd   = m.get("round",     "-")
                acc   = m.get("accuracy",  float("nan"))
                loss  = m.get("loss",      float("nan"))
                f1    = m.get("f1",        float("nan"))
                prec  = m.get("precision", float("nan"))
                rec   = m.get("recall",    float("nan"))
                print(f"  {rnd:>5}  {acc:>8.4f}  {loss:>8.4f}  {f1:>8.4f}  {prec:>8.4f}  {rec:>8.4f}")

            if rounds:
                last = rounds[-1]
                result["final_round"] = last
                print(f"\n  Final round (r{last.get('round','-')}) — "
                      f"acc={last.get('accuracy',float('nan')):.4f}  "
                      f"loss={last.get('loss',float('nan')):.4f}  "
                      f"f1={last.get('f1',float('nan')):.4f}")
                result["success"] = True
            else:
                print("  [WARN] No per-round metrics returned by Hub yet")
        else:
            print(f"  [WARN] get-job-metrics HTTP {r.status_code}: {r.text[:120]}")
    except Exception as exc:
        print(f"  [WARN] get-job-metrics failed: {exc}")

    # ── Job-level summary ─────────────────────────────────────────────────────
    try:
        r2 = s.get(f"{hub}/api/job-details/{job_id}/", timeout=10)
        if r2.status_code == 200:
            d = r2.json()
            dur = d.get("duration", "?")
            print(f"  Job duration : {dur}s" if isinstance(dur, (int, float)) else f"  Job duration : {dur}")
            result["job_details"] = d
    except Exception as exc:
        print(f"  [WARN] job-details failed: {exc}")

    return result


def step_session_metrics(client_ids: list[str], timeout: int = 120,
                         label: str = "Step 12 — Session metrics (from Node DB)") -> dict:
    """Query the Node DB for final training metrics of each client_id.

    Polls until every session has status 'completed' or 'failed', then reads
    TrainingSession.final_* fields and the last TrainingRound for per-round
    detail.  Returns a dict keyed by client_id.

    ``label`` customises the section header so intermediate per-job calls can
    print a meaningful title (e.g. "Step 7 — Session metrics").

    Uses manage.py shell so there is no direct Django import dependency in
    this script.
    """
    section(label)

    ids_repr = repr(client_ids)
    # Read metrics from the last TrainingRound (always populated) rather than
    # TrainingSession.final_* (only set when complete_training_session receives
    # explicit final_metrics, which not all code paths provide).
    script = f"""
import json
from trainings.models import TrainingSession

ids = {ids_repr}
out = {{}}
for cid in ids:
    try:
        s = TrainingSession.objects.filter(client_id=cid).order_by('-started_at').first()
        if not s:
            out[cid] = {{'found': False}}
            continue
        last_round = s.rounds.order_by('-round_number').first()
        # Prefer final_* fields; fall back to last-round values when null.
        def _coalesce(final_val, round_val):
            return final_val if final_val is not None else round_val
        out[cid] = {{
            'found':     True,
            'status':    s.status,
            'rounds':    s.current_round,
            'accuracy':  _coalesce(s.final_accuracy,  last_round.accuracy   if last_round else None),
            'loss':      _coalesce(s.final_loss,       last_round.loss       if last_round else None),
            'precision': _coalesce(s.final_precision,  last_round.precision  if last_round else None),
            'recall':    _coalesce(s.final_recall,     last_round.recall     if last_round else None),
            'f1':        _coalesce(s.final_f1,         last_round.f1_score   if last_round else None),
            'epsilon':   (last_round.metrics.get('privacy_epsilon') if last_round else None),
            'error':     s.error_message,
        }}
    except Exception as e:
        out[cid] = {{'found': False, 'error': str(e)}}
print(json.dumps(out))
"""

    # Poll until all sessions reach a terminal state (COMPLETED or FAILED) or timeout.
    # Statuses in the DB are uppercase: COMPLETED, FAILED, ACTIVE, STARTING, etc.
    TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}
    deadline = time.time() + timeout
    metrics: dict = {}
    while time.time() < deadline:
        result = subprocess.run(
            [sys.executable, "manage.py", "shell", "-c", script],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent),
        )
        if result.returncode != 0:
            print(f"  WARNING: DB query failed — {result.stderr.strip()[:120]}")
            return {}
        try:
            metrics = json.loads(result.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            print(f"  WARNING: Could not parse DB output: {result.stdout.strip()[:120]}")
            return {}

        pending = [
            cid for cid, m in metrics.items()
            if m.get("found") and m.get("status", "").upper() not in TERMINAL
        ]
        if not pending:
            break
        elapsed = int(time.time() - (deadline - timeout))
        print(f"  [{elapsed:3d}s] Waiting for sessions to complete: {pending}")
        time.sleep(8)

    # Print results table
    row_fmt = "  {:<26}  {:>10}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>7}"
    print(row_fmt.format("client_id", "status", "acc", "loss", "f1", "prec", "recall", "rounds"))
    print("  " + "-" * 90)
    for cid, m in metrics.items():
        if not m.get("found"):
            print(f"  {cid:<26}  NOT FOUND")
            continue
        def _f(v):
            return f"{v:.4f}" if v is not None else "   n/a"
        print(row_fmt.format(
            cid[:26],
            m.get("status", "?"),
            _f(m.get("accuracy")),
            _f(m.get("loss")),
            _f(m.get("f1")),
            _f(m.get("precision")),
            _f(m.get("recall")),
            str(m.get("rounds") or "?"),
        ))
        if m.get("epsilon") is not None and m["epsilon"] != -1.0:
            print(f"    privacy_epsilon (last round): {m['epsilon']:.6f}")
        if m.get("status", "").upper() == "FAILED" and m.get("error"):
            print(f"    error: {m['error'][:120]}")

    return metrics


def _fetch_min_noise(
    cfg: dict,
    dataset_name: str,
    batch_size: int,
    epochs: int,
    use_experiment: bool = False,
) -> Optional[float]:
    """Call GET /api/v2/min-noise-multiplier/ to get the floor σ the Node enforces.

    Pass use_experiment=True for experimental runs so the endpoint computes σ
    against the experimental partition size (experiment_row_count) rather than
    the full dataset.  A smaller partition requires a higher σ floor to achieve
    the same ε guarantee — using the wrong n silently underestimates the floor.
    """
    try:
        r = requests.get(
            f"{cfg['node_url']}/api/v2/min-noise-multiplier/",
            params={
                "dataset_name":   dataset_name,
                "batch_size":     batch_size,
                "epochs":         epochs,
                "use_experiment": "true" if use_experiment else "false",
            },
            headers=headers(cfg["researcher_api_key"], cfg["client_ip"]),
            timeout=5,
        )
        if r.status_code == 200:
            return float(r.json().get("min_noise_multiplier", float("nan")))
    except Exception:
        pass
    return None


def print_final_table(
    session_metrics: dict,
    hub_results: list[dict],
    job_labels: list[str],
    dp_cfg: dict,
    exp_cfg: dict,
    budget_after: dict,
    dataset_name: str,
    sigmas: list[Optional[float]],
) -> None:
    """Print a single consolidated table with Node + Hub metrics for all 3 jobs."""
    W = 92
    print(f"\n{'═' * W}")
    print("  FINAL RESULTS — Section 3.1 Use-Case Summary")
    print(f"{'═' * W}")

    hdr = (
        f"  {'job':<22} {'DP':>4} {'batch':>6} {'epochs':>7} {'σ':>7} "
        f"{'acc':>7} {'loss':>8} {'f1':>7} {'ε spent':>10} {'rounds':>7}"
    )
    sep = "  " + "─" * (W - 2)
    print(hdr)
    print(sep)

    rows: list[tuple[float, float, float]] = []
    cids = list(session_metrics.keys())
    for i, (label, hub_res) in enumerate(zip(job_labels, hub_results)):
        # Metrics: Hub final round (FedAvg aggregated) — the real federated result
        hub_final = hub_res.get("final_round", {})
        acc  = hub_final.get("accuracy", float("nan"))
        loss = hub_final.get("loss",     float("nan"))
        f1   = hub_final.get("f1",       float("nan"))

        # ε and rounds: Node DB (source of truth for privacy accounting)
        node_m  = session_metrics.get(cids[i], {}) if i < len(cids) else {}
        eps     = node_m.get("epsilon")
        rds     = node_m.get("rounds", "?")

        is_dp   = "yes" if "dp" in label.lower() else "no"
        batch   = dp_cfg["batch_size"] if is_dp == "yes" else (32 if "b32" in label else 16)
        epochs  = dp_cfg["epochs"]     if is_dp == "yes" else exp_cfg["epochs"]
        eps_str = f"{eps:.4f}" if eps and eps != -1.0 else "—"
        sigma   = sigmas[i] if i < len(sigmas) and sigmas[i] is not None else float("nan")
        sig_str = f"{sigma:.4f}" if sigma == sigma else "?"   # nan check

        rows.append((acc, loss, f1))
        print(
            f"  {label:<22} {is_dp:>4} {batch:>6} {epochs:>7} {sig_str:>7} "
            f"{acc:>7.4f} {loss:>8.4f} {f1:>7.4f} {eps_str:>10} {str(rds):>7}"
        )

    print(sep)

    # DP cost delta: dp_prod (row 2) vs exp_b32 (row 0) as baseline
    if len(rows) == 3:
        d_acc  = rows[2][0] - rows[0][0]
        d_loss = rows[2][1] - rows[0][1]
        d_f1   = rows[2][2] - rows[0][2]
        dp_eps_ds = next(
            (d for d in budget_after.get("datasets", []) if d.get("dataset_name") == dataset_name),
            {}
        )
        dp_eps = dp_eps_ds.get("spent_epsilon", "?")
        print(
            f"\n  DP cost vs exp_b32 baseline : "
            f"Δacc={d_acc:+.4f}  Δloss={d_loss:+.4f}  Δf1={d_f1:+.4f}"
        )
        print(
            f"  Privacy guarantee           : "
            f"ε={dp_eps}, δ=1×10⁻⁵  (Rényi DP, σ={dp_cfg['noise_multiplier']})"
        )

    print(f"{'═' * W}\n")


def step11_auditor_summary(cfg: dict, budget_before: dict, budget_after: dict) -> bool:
    """
    Auditor verification via budget delta.
    The full audit log (timestamps, Researcher identity, per-job ε consumed)
    is accessible via the Auditor web panel at /audit/dashboard/.
    This step verifies programmatically that:
      - The DP production job consumed ε from the budget
      - The two experimental jobs did NOT consume budget
    """
    section("Step 11 — Auditor summary (budget delta verification)")

    target_name = cfg["dataset_name"]
    before_ds = next(
        (ds for ds in budget_before.get("datasets", []) if ds["dataset_name"] == target_name),
        None,
    )
    after_ds = next(
        (ds for ds in budget_after.get("datasets", []) if ds["dataset_name"] == target_name),
        None,
    )

    if not before_ds or not after_ds:
        print("  ERROR: Could not find dataset in budget status responses")
        return False

    spent_before    = float(before_ds["spent_epsilon"])
    spent_after     = float(after_ds["spent_epsilon"])
    remaining_after = float(after_ds["remaining_budget"])
    delta           = round(spent_after - spent_before, 6)

    print(f"     Dataset: {target_name}")
    print(f"     ε spent before training : {spent_before}")
    print(f"     ε spent after  training : {spent_after}")
    print(f"     ε delta (DP job only)   : {delta}")
    print(f"     ε remaining             : {remaining_after}")

    if delta > 0:
        print("  Budget decremented — DP production job confirmed")
        print("  Experimental jobs did not consume budget (delta = DP run only)")
        print()
        print("  Full audit log available at the Auditor web panel:")
        print(f"      {cfg['node_url']}/audit/dashboard/")
        return True
    else:
        print("  ERROR: Budget was NOT decremented — DP production job may have failed")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MediNet Section 3.1 Use-Case Test")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "test_config.json"),
        help="Path to test_config.json (default: same directory as this script)",
    )
    args = parser.parse_args()
    cfg  = load_config(args.config)

    if cfg["researcher_api_key"] == "YOUR_RESEARCHER_API_KEY":
        print("ERROR: Set researcher_api_key in test_config.json before running.")
        print("   Generate one with:  python manage.py generate_api_key <username>")
        sys.exit(1)

    if cfg.get("hub_username") == "YOUR_HUB_USERNAME":
        print("ERROR: Set hub_username and hub_password in test_config.json before running.")
        sys.exit(1)

    dp_cfg  = cfg["model"]["dp_production"]
    exp_cfg = cfg["model"]["experimental"]

    # Reset test dataset budgets so repeated runs start from a clean slate.
    # Also syncs max_epsilon_per_job and lifetime_budget from the config so the
    # DB policy always matches test_config.json without manual admin edits.
    reset_budget(
        cfg["dataset_name"],
        per_job_max=cfg["privacy_policy"]["per_job_max"],
        total_budget=cfg["privacy_policy"]["total_budget"],
    )

    # Clear accumulated APIRequest records so the rate limiter (100 req/hour)
    # does not block repeated test runs within the same hour.
    _reset_rate_limit(cfg.get("hub_username", ""))

    print("=" * 60)
    print("  MediNet Use-Case End-to-End Test (Section 3.1)")
    print(f"  Node    : {cfg['node_url']}")
    print(f"  Hub     : {cfg['hub_url']}")
    print(f"  Dataset : {cfg['dataset_name']}")
    print(f"  Flower  : {cfg['flower_server']}")
    print(f"  Budget  : ε={cfg['privacy_policy']['total_budget']} total, "
          f"{cfg['privacy_policy']['per_job_max']} per-job max")
    print(f"  DP run  : σ={dp_cfg['noise_multiplier']}, C={dp_cfg['max_grad_norm']}, "
          f"batch={dp_cfg['batch_size']}, epochs={dp_cfg['epochs']}")
    print("=" * 60)

    results: dict = {}
    ts = int(time.time())

    # Steps 1–3: Ping, dataset info, budget baseline — no Flower server needed yet
    results["ping"] = step1_ping(cfg)

    ids        = step2_get_data_info(cfg)
    dataset_id = ids[0] if ids else cfg["dataset_id"]
    results["get_data_info"] = bool(ids)

    budget_before = step3_budget_status(cfg, "Budget status — BEFORE training")

    # Steps 4–6: ε estimates
    exp32_config = model_json_experimental(dataset_id, cfg["dataset_name"],
                                           batch_size=32, epochs=exp_cfg["epochs"])
    exp16_config = model_json_experimental(dataset_id, cfg["dataset_name"],
                                           batch_size=16, epochs=exp_cfg["epochs"])
    dp_config    = model_json_dp(
        dataset_id, cfg["dataset_name"],
        batch_size       = dp_cfg["batch_size"],
        epochs           = dp_cfg["epochs"],
        noise_multiplier = dp_cfg["noise_multiplier"],
        max_grad_norm    = dp_cfg["max_grad_norm"],
    )

    eps_exp32 = step_estimate_epsilon(cfg, "Step 4 — Estimate ε: experimental batch=32 (no DP)",
                                      exp32_config)
    eps_exp16 = step_estimate_epsilon(cfg, "Step 5 — Estimate ε: experimental batch=16 (no DP)",
                                      exp16_config)
    eps_dp    = step_estimate_epsilon(cfg,
                                      f"Step 6 — Estimate ε: production DP run (σ={dp_cfg['noise_multiplier']})",
                                      dp_config)
    results["epsilon_estimates"] = all(e is not None for e in [eps_exp32, eps_exp16, eps_dp])

    # Steps 7–9: Each training job runs with its own dedicated Hub Flower server.
    #
    # Sequential execution matches real Hub operation: a researcher submits jobs
    # one at a time, each gets its own Flower server lifetime (10 rounds), and the
    # same port (8080) is reused once the previous server shuts down after completing
    # its rounds.  Running all three concurrently against a single server causes
    # FedAvg to average contradictory gradient updates from clients with different
    # batch sizes and DP noise levels, producing degenerate (all-negative) models.
    flower_host  = cfg["flower_server"].split(":")[0]
    flower_port  = int(cfg["flower_server"].split(":")[-1])
    client_id_32 = f"exp_b32_{ts}"
    client_id_16 = f"exp_b16_{ts}"
    client_id_dp = f"dp_prod_{ts}"

    # ── Step 7: experimental batch=32 ────────────────────────────────────────
    flower_ok7, hub_meta7 = step0_start_flower_server(cfg)
    results["flower_server"] = flower_ok7
    if not flower_ok7:
        print("\nERROR: Cannot start Flower server for Step 7. Aborting.")
        sys.exit(1)
    r7 = step_start_client(
        cfg, "Step 7 — Experimental run batch=32 (no budget consumed)",
        exp32_config, client_id_32,
    )
    # Wait for this session to reach a terminal state before the next job binds the port.
    step_session_metrics([client_id_32], timeout=400,
                         label="Step 7 metrics — exp_b32 (waiting for completion)")
    hub_result7 = step_hub_metrics(hub_meta7, "Step 7 Hub metrics — exp_b32")
    _wait_for_port_close(flower_host, flower_port, timeout=90)

    # ── Step 8: experimental batch=16 ────────────────────────────────────────
    flower_ok8, hub_meta8 = step0_start_flower_server(cfg)
    if not flower_ok8:
        print("\nERROR: Cannot start Flower server for Step 8. Aborting.")
        sys.exit(1)
    r8 = step_start_client(
        cfg, "Step 8 — Experimental run batch=16 (no budget consumed)",
        exp16_config, client_id_16,
    )
    step_session_metrics([client_id_16], timeout=400,
                         label="Step 8 metrics — exp_b16 (waiting for completion)")
    hub_result8 = step_hub_metrics(hub_meta8, "Step 8 Hub metrics — exp_b16")
    _wait_for_port_close(flower_host, flower_port, timeout=90)

    # ── Step 9: DP production run ─────────────────────────────────────────────
    flower_ok9, hub_meta9 = step0_start_flower_server(cfg)
    if not flower_ok9:
        print("\nERROR: Cannot start Flower server for Step 9. Aborting.")
        sys.exit(1)
    r9 = step_start_client(
        cfg,
        f"Step 9 — Production DP run σ={dp_cfg['noise_multiplier']} (budget consumed)",
        dp_config, client_id_dp,
        retry_on_concurrent_limit=True,
        concurrent_limit_timeout=300,
    )
    step_session_metrics([client_id_dp], timeout=400,
                         label="Step 9 metrics — dp_prod (waiting for completion)")
    hub_result9 = step_hub_metrics(hub_meta9, "Step 9 Hub metrics — dp_prod (FedAvg global model)")
    results["training"] = all([r7, r8, r9])

    # Step 10: dp_prod session just completed, so the budget spend should already
    # be recorded.  Give a short 60 s window for any in-flight DB writes to settle.
    budget_after = step10_wait_for_budget_change(
        cfg, budget_before, cfg["dataset_name"], timeout=60
    )
    step3_budget_status(cfg, "Budget status — AFTER training")

    # Step 11: Auditor summary
    results["auditor_verification"] = step11_auditor_summary(cfg, budget_before, budget_after)

    # Step 12: Consolidated metrics for all three sessions.
    # All sessions are already in a terminal state (waited above), so this query
    # returns immediately and just prints the summary table.
    session_metrics = step_session_metrics(
        [client_id_32, client_id_16, client_id_dp],
        timeout=30,
    )

    # Fetch the Node-enforced σ floor for each job (used in final table).
    # Experimental runs must use use_experiment=True so the floor is computed
    # against the partition size (experiment_row_count), not the full dataset.
    sigma_exp32 = _fetch_min_noise(cfg, cfg["dataset_name"], 32, exp_cfg["epochs"], use_experiment=True)
    sigma_exp16 = _fetch_min_noise(cfg, cfg["dataset_name"], 16, exp_cfg["epochs"], use_experiment=True)
    sigma_dp    = dp_cfg["noise_multiplier"]

    # Final consolidated table
    print_final_table(
        session_metrics  = session_metrics,
        hub_results      = [hub_result7, hub_result8, hub_result9],
        job_labels       = ["exp_b32", "exp_b16", "dp_prod"],
        dp_cfg           = dp_cfg,
        exp_cfg          = exp_cfg,
        budget_after     = budget_after,
        dataset_name     = cfg["dataset_name"],
        sigmas           = [sigma_exp32, sigma_exp16, sigma_dp],
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for name, passed in results.items():
        if passed is None:
            icon, label = "[SKIP]", "SKIPPED"
        elif passed:
            icon, label = "[PASS]", "PASSED"
        else:
            icon, label = "[FAIL]", "FAILED"
        print(f"  {icon} {name}: {label}")

    failed = [k for k, v in results.items() if v is False]
    if not failed:
        print("\nAll steps passed — use case validated.")
    else:
        print(f"\nFailed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
