# MediNet Core — Developer Reference

`medinet_core` is the shared abstractions package that every MediNet deployment
inherits.  It provides the security layer, role system, upload pipeline, and
base models that keep domain-specific forks (EpiFlare, OncaNet, …) consistent
without duplicating code.

---

## Table of Contents

1. [Package layout](#1-package-layout)
2. [Security middleware](#2-security-middleware)
3. [Role system](#3-role-system)
4. [Uploader framework](#4-uploader-framework)
5. [Base models](#5-base-models)
6. [Settings pattern](#6-settings-pattern)
7. [How to create a fork](#7-how-to-create-a-fork)
8. [Extension guide](#8-extension-guide)
   - [Adding a new role](#81-adding-a-new-role)
   - [Adding a new uploader](#82-adding-a-new-uploader)
   - [Adding a new middleware](#83-adding-a-new-middleware)
   - [Adding a new domain app](#84-adding-a-new-domain-app)

---

## 1. Package layout

```
medinet_core/
├── apps.py                     # AppConfig — registers base roles on startup
├── models/
│   └── upload.py               # BaseUploadRecord — abstract model
├── roles/
│   ├── registry.py             # RoleRegistry + RoleDefinition dataclass
│   └── base_roles.py           # ADMIN, MEMBER, RESEARCHER, AUDITOR definitions
├── security/
│   └── middleware.py           # 4 middleware classes
├── templatetags/
│   └── csp_tags.py             # {% csp_nonce %} template tag
└── uploaders/
    ├── base.py                 # BaseUploader abstract class + UploadError
    └── registry.py             # UploaderRegistry singleton
```

---

## 2. Security middleware

All four classes live in `medinet_core/security/middleware.py` and are active
in every deployment through `config/settings/base.py`.

### Middleware stack (in order)

```python
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'medinet_core.security.middleware.SecurityHeadersMiddleware',   # 1
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'medinet_core.security.middleware.SessionTimeoutMiddleware',    # 2
    'medinet_core.security.middleware.APIAuthenticationMiddleware', # 3
    'medinet_core.security.middleware.RateLimitMiddleware',         # 4
    'audit.middleware.AuditMiddleware',
    ...
]
```

### SecurityHeadersMiddleware

Generates a unique **CSP nonce** per request and attaches it to
`request.csp_nonce`.  Sets the following response headers on every response:

| Header | Value |
|---|---|
| `Content-Security-Policy` | nonce-based; no `unsafe-inline` |
| `X-Content-Type-Options` | `nosniff` |
| `Referrer-Policy` | `strict-origin-when-cross-origin` |
| `Permissions-Policy` | geolocation, mic, camera, payment, usb disabled |
| `Server` | removed |

Templates access the nonce through the `{% csp_nonce %}` tag:

```html
{% load csp_tags %}
<script nonce="{% csp_nonce %}">...</script>
<style  nonce="{% csp_nonce %}">...</style>
```

### SessionTimeoutMiddleware

Auto-logs out authenticated users after `SESSION_IDLE_TIMEOUT` seconds of
inactivity (default: 7200 s / 2 h, configurable per deployment).  AJAX
requests receive a `401` JSON response instead of a redirect.

Also enforces that **RESEARCHER** users can only access `/api/v2/` and
`/info/researcher` — any other path is blocked and logged as a security
violation.

### APIAuthenticationMiddleware

Stateless API key authentication for requests to `/api/`.

- Reads `X-API-Key` header.
- Extracts `key_prefix` (first 8 characters) for an **O(1)** DB candidate
  narrowing before checking the full key hash.
- Only RESEARCHER-role keys are accepted for API access.
- Sets `request.api_user` on success; returns `401` JSON on failure.
- All authentication attempts are logged to the `security` logger.

### RateLimitMiddleware

Per-user (or per-IP for anonymous) rate limiting using Django's cache backend.
Settings keys: `RATELIMIT_ENABLE` (bool), `RATELIMIT_USE_CACHE` (cache alias).

---

## 3. Role system

### Concepts

A **role** is a named set of permissions stored as a JSON field on the `Role`
model.  Each `CustomUser` has a FK to one `Role`.  Permission checks happen at
the model level via `user.has_permission(key, domain=None)`.

### Permission value types

```python
# Boolean — user either has it or doesn't
'audit.view': True

# Scoped — user has it, but only for specific domains
'inference.execute': {'scope': 'ALL'}            # any domain
'inference.execute': {'scope': ['cardiology']}   # explicit list
```

When `has_permission(key)` is called **without** a domain it returns `True` if
the permission exists at all (regardless of scope).  When called **with** a
domain it enforces the scope list.

### Base roles (MediNetNode)

| Role | Description |
|---|---|
| `ADMIN` | Full access — user management, system settings, all inference |
| `MEMBER` | Clinical staff — dataset upload/train, inference execution |
| `RESEARCHER` | External API-only access — stateless, no web UI |
| `AUDITOR` | Read-only — audit logs, datasets, trainings, users |

### RoleRegistry

`role_registry` is a module-level singleton imported from
`medinet_core.roles.registry`.

```python
from medinet_core.roles.registry import role_registry, RoleDefinition

# Read all registered roles
for role_def in role_registry.all():
    print(role_def.name, role_def.permissions)

# Check if a role exists
if 'GENETICIST' in role_registry:
    ...

# Django choices for model fields
Role.objects.values_list('name', flat=True)  # from DB
role_registry.as_role_choices()              # from registry (pre-DB)
```

### setup_roles management command

The `setup_roles` command reads from `role_registry.all()` and creates or
updates `Role` objects in the database.  Because all role definitions go
through the registry, **no command changes are needed when adding a new role**
— just register it in `ready()`.

```bash
python manage.py setup_roles          # create missing roles
python manage.py setup_roles --force  # overwrite all permissions
```

---

## 4. Uploader framework

### How it works

1. A domain `AppConfig.ready()` registers one or more `BaseUploader` subclasses
   with the `uploader_registry` singleton.
2. A view retrieves the correct uploader by file extension and delegates the
   entire upload pipeline to it.
3. The uploader handles validation, checksums, business logic, and persistence.

### BaseUploader contract

```python
from medinet_core.uploaders.base import BaseUploader, UploadError

class MyUploader(BaseUploader):

    # Required: map extension → list of accepted MIME types
    ALLOWED_EXTENSIONS = {
        '.csv': ['text/csv', 'text/plain', 'application/csv'],
    }

    def upload(self, file_path: str, **kwargs):
        """
        Must return (domain_model_instance, info_dict).
        Raise UploadError on failure.
        """
        self._validate_file(file_path)          # extension + MIME check
        checksum = self._calculate_checksum(file_path)
        self._update_progress('parsing', 'Reading file...')
        # ... domain-specific logic ...
        return model_instance, {'rows': n}
```

### Shared utilities provided by BaseUploader

| Method | Purpose |
|---|---|
| `_validate_file(path)` | Extension + MIME-type check (uses `python-magic` when available) |
| `_calculate_checksum(path)` | SHA-256 hex digest |
| `_update_progress(stage, msg)` | Calls `progress_callback` if provided |
| `_get_quarantine_dir()` | Returns `MEDIA_ROOT/quarantine/` (created on demand) |
| `_is_phi_column(name)` | Returns `True` if column name matches a PHI pattern |
| `_make_temp_dir()` | Creates a `tempfile.mkdtemp` working directory |

### Platform-wide constants

```python
BaseUploader.MIN_K_ANONYMITY   # = 5  — minimum rows for k-anonymity
BaseUploader.FORBIDDEN_PATTERNS  # PHI column patterns (id, ssn, name, dob…)
```

### UploaderRegistry

```python
from medinet_core.uploaders.registry import uploader_registry

# Look up by extension (returns None if not registered)
uploader_cls = uploader_registry.get('.csv')
if uploader_cls is None:
    raise Http404('Unsupported file type')

uploader = uploader_cls(request.user, progress_callback=my_cb)
result, info = uploader.upload(tmp_path, batch=batch, ...)

# Inspect registered extensions
uploader_registry.extensions()  # ['.csv', '.idat', ...]
```

### View-side pattern (background upload)

```python
def _process_upload_background(batch_id, user_id, file_path, **kwargs):
    close_old_connections()  # required in worker threads
    try:
        user  = CustomUser.objects.get(pk=user_id)
        batch = UploadBatch.objects.get(id=batch_id)
        uploader_cls = uploader_registry.get('.csv')
        uploader = uploader_cls(user)
        uploader.upload(file_path, batch=batch, **kwargs)
    except Exception as exc:
        UploadBatch.objects.filter(id=batch_id).update(
            status='failed', error_message=str(exc)
        )
```

---

## 5. Base models

### BaseUploadRecord

Abstract model in `medinet_core/models/upload.py`.  Inherit from it in any
domain model that tracks uploaded files.

```python
from medinet_core.models import BaseUploadRecord

class GeneticsSample(BaseUploadRecord):
    patient   = models.ForeignKey(Patient, on_delete=models.CASCADE)
    probe_count = models.IntegerField()

    class Meta(BaseUploadRecord.Meta):
        app_label = 'genetics'
```

**Fields provided:**

| Field | Type | Notes |
|---|---|---|
| `file_path` | `CharField(500)` | Absolute path on disk |
| `file_size` | `BigIntegerField` | Bytes |
| `checksum_sha256` | `CharField(64)` | SHA-256, editable=False |
| `uploaded_by_id` | `IntegerField` | PK of `CustomUser` — plain int (cross-DB safe) |
| `uploaded_at` | `DateTimeField` | auto_now_add |
| `is_active` | `BooleanField` | soft-delete flag |

**Shared methods:**

```python
instance.calculate_checksum()       # recomputes SHA-256 from disk
instance.get_file_size_display()    # "4.2 MB"
instance.ensure_checksum()          # sets checksum_sha256 if not already set
```

> **Why `uploaded_by_id` is a plain int**: the platform uses two SQLite
> databases (`users_logs` and `datasets_db`).  Cross-database foreign keys are
> not supported by Django's ORM, so ownership is stored as a bare integer
> rather than a FK constraint.

---

## 6. Settings pattern

### Three-layer hierarchy

```
config/settings/base.py          ← shared across all deployments
config/settings/<deployment>.py  ← deployment-specific overrides
config/settings/test.py          ← test environment (in-memory DBs, fast hashers)
```

`base.py` defines everything that every deployment shares:
- `BASE_APPS` list (all shared Django apps)
- Full `MIDDLEWARE` stack
- `DATABASES` structure (default + datasets_db)
- Auth, session, CSRF, cache, logging settings
- `ROOT_URLCONF = None` — **must** be overridden by each deployment

A deployment settings file is minimal:

```python
# config/settings/medinet.py
from .base import *

ROOT_URLCONF       = 'medinet.urls'
WSGI_APPLICATION   = 'medinet.wsgi.application'
handler403         = 'medinet.error_handlers.handler403'
# any deployment-specific overrides…
```

A fork adds its extra apps:

```python
# config/settings/epiflare.py
from config.settings.base import *

INSTALLED_APPS = BASE_APPS + ['patients', 'genetics']
ROOT_URLCONF   = 'epiflare.urls'
WSGI_APPLICATION = 'epiflare.wsgi.application'
```

### Backward-compatibility wrappers

The old `<project>/settings.py` and `<project>/settings_test.py` files are
kept as thin wrappers so legacy scripts and tools continue to work without
modification:

```python
# medinet/settings.py
from config.settings.medinet import *  # noqa: F401, F403
```

---

## 7. How to create a fork

A fork is a new Django project that uses `medinet_core` as its shared package
and adds domain-specific apps.

### Step-by-step

```bash
# 1. Clone MediNetNode
git clone <medinet-node-repo> my-deployment
cd my-deployment
git remote rename origin upstream

# 2. Create your deployment settings
# config/settings/mydeployment.py
from config.settings.base import *
INSTALLED_APPS = BASE_APPS + ['myapp']
ROOT_URLCONF   = 'mydeployment.urls'
WSGI_APPLICATION = 'mydeployment.wsgi.application'

# 3. Wire entry points
# manage.py  →  os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.mydeployment')
# wsgi.py    →  same
# conftest.py →  os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.test')

# 4. Create test settings
# config/settings/test.py
from config.settings.mydeployment import *
DATABASES = {'default': {'ENGINE': '...sqlite3', 'NAME': ':memory:'}, ...}
PASSWORD_HASHERS = ['django.contrib.auth.hashers.MD5PasswordHasher']

# 5. Add your domain app(s), register roles and uploaders
# 6. Run migrations
python manage.py migrate
python manage.py setup_roles
```

### What you inherit for free

- All security middleware (CSP, session timeout, API auth, rate limiting)
- Role/permission system with `has_permission()` and scope support
- Uploader registry pattern (just subclass and register)
- Audit logging (`audit.middleware.AuditMiddleware`)
- Dual-DB router (`core.routers.DatabaseRouter`)
- API key authentication for RESEARCHER-type roles
- All 20+ templates with CSP nonce already in place

---

## 8. Extension guide

### 8.1 Adding a new role

**When**: your fork needs a user type not present in MediNetNode (e.g.
`RADIOLOGIST`, `ONCOLOGIST`, `GENETICIST`).

**Where to add it**: in your fork's copy of
`medinet_core/roles/base_roles.py`, or in your domain `AppConfig.ready()`.

**Option A — add to base_roles.py** (simpler, role visible to `setup_roles`):

```python
# medinet_core/roles/base_roles.py
_BASE_ROLES = [
    # … existing roles …
    RoleDefinition(
        name='RADIOLOGIST',
        display_name='Radiologist',
        description='Access to radiology datasets and inference.',
        permissions={
            'api.access': True,
            'dataset.view': {'scope': ['radiology']},
            'inference.execute': {'scope': ['radiology']},
            'radiology.upload': True,
        },
    ),
]
```

**Option B — register in AppConfig.ready()** (preferred for fork-specific roles):

```python
# myapp/apps.py
from django.apps import AppConfig

class MyAppConfig(AppConfig):
    name = 'myapp'

    def ready(self):
        from medinet_core.roles.registry import role_registry, RoleDefinition
        role_registry.register(RoleDefinition(
            name='RADIOLOGIST',
            display_name='Radiologist',
            permissions={
                'api.access': True,
                'dataset.view': {'scope': ['radiology']},
                'inference.execute': {'scope': ['radiology']},
            },
        ))
```

Then run:

```bash
python manage.py setup_roles --force
```

**Permission naming convention**: use `<domain>.<action>` dot-notation.

---

### 8.2 Adding a new uploader

**When**: your fork handles a new file format (IDAT, VCF, DICOM, …).

**1. Create the uploader class:**

```python
# myapp/uploaders.py
from medinet_core.uploaders.base import BaseUploader, UploadError

class IDATUploader(BaseUploader):
    ALLOWED_EXTENSIONS = {
        '.idat': ['application/octet-stream'],
    }

    def upload(self, file_path, *, batch, sample_name, **kwargs):
        self._validate_file(file_path)
        checksum = self._calculate_checksum(file_path)
        self._update_progress('parsing', 'Reading IDAT file…')

        # domain-specific processing
        try:
            data = self._parse_idat(file_path)
        except Exception as exc:
            raise UploadError(f"IDAT parse failed: {exc}") from exc

        # persist
        record = IDATRecord.objects.create(
            file_path=file_path,
            file_size=os.path.getsize(file_path),
            checksum_sha256=checksum,
            uploaded_by_id=self.user.pk,
            batch=batch,
            sample_name=sample_name,
            probe_count=len(data),
        )
        self._update_progress('done', f'Imported {len(data)} probes.')
        return record, {'probe_count': len(data)}

    def _parse_idat(self, file_path):
        # your parsing logic
        ...
```

**2. Register in AppConfig.ready():**

```python
# myapp/apps.py
class MyAppConfig(AppConfig):
    name = 'myapp'

    def ready(self):
        from medinet_core.uploaders.registry import uploader_registry
        from myapp.uploaders import IDATUploader
        uploader_registry.register(IDATUploader)
```

**3. Use in a view:**

```python
from medinet_core.uploaders.registry import uploader_registry

uploader_cls = uploader_registry.get('.idat')
uploader = uploader_cls(request.user)
record, info = uploader.upload(tmp_path, batch=batch, sample_name=name)
```

---

### 8.3 Adding a new middleware

**When**: your fork needs a cross-cutting concern not present in the core (e.g.
tenant detection, feature flags, custom rate limiting per endpoint).

Custom middleware belongs in your fork's own package, not in `medinet_core`
(unless it is truly platform-wide and you intend to contribute it back).

```python
# myapp/middleware.py
class TenantMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # resolve tenant from subdomain or header
        request.tenant = self._resolve_tenant(request)
        return self.get_response(request)

    def _resolve_tenant(self, request):
        ...
```

Add it to your deployment settings **after** the core middleware stack:

```python
# config/settings/mydeployment.py
from config.settings.base import *

MIDDLEWARE = list(MIDDLEWARE) + [
    'myapp.middleware.TenantMiddleware',
]
```

If the middleware needs to run **before** session handling, insert it
explicitly:

```python
from config.settings.base import MIDDLEWARE as _BASE

_idx = _BASE.index('django.contrib.sessions.middleware.SessionMiddleware')
MIDDLEWARE = list(_BASE)
MIDDLEWARE.insert(_idx, 'myapp.middleware.TenantMiddleware')
```

---

### 8.4 Adding a new domain app

A domain app is a standard Django app that contains the models, views, URLs,
and business logic specific to your fork.

**Minimal checklist:**

```
myapp/
├── __init__.py
├── apps.py            ← AppConfig with ready() for registry hooks
├── models.py          ← inherit BaseUploadRecord where appropriate
├── views.py
├── urls.py            ← app_name = 'myapp'
├── uploaders.py       ← BaseUploader subclass(es)
├── templates/
│   └── myapp/
│       └── dashboard.html   ← {% load csp_tags %} + nonce on all <script>/<style>
└── migrations/
```

**apps.py template:**

```python
from django.apps import AppConfig

class MyAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'myapp'

    def ready(self):
        from medinet_core.roles.registry import role_registry, RoleDefinition
        from medinet_core.uploaders.registry import uploader_registry
        from myapp.uploaders import MyUploader

        role_registry.register(RoleDefinition(
            name='MY_ROLE',
            permissions={'myapp.view': True, 'myapp.upload': True},
        ))
        uploader_registry.register(MyUploader)
```

**Wire into the deployment:**

```python
# config/settings/mydeployment.py
INSTALLED_APPS = BASE_APPS + ['myapp']
```

```python
# mydeployment/urls.py
urlpatterns = [
    ...
    path('myapp/', include('myapp.urls')),
]
```

**Template CSP requirement** — every `<script>` and `<style>` tag must carry
the nonce, otherwise the browser will block it:

```html
{% load csp_tags %}
<!DOCTYPE html>
<html>
<head>
  <style nonce="{% csp_nonce %}">
    /* inline styles */
  </style>
</head>
<body>
  <script nonce="{% csp_nonce %}">
    // inline scripts
  </script>
</body>
</html>
```

External CDN resources (`src=` or `href=`) do not need a nonce — they are
already allowed by the `script-src 'self' https://cdn.jsdelivr.net` directive
in `SecurityHeadersMiddleware`.

---

## Quick reference

```python
# Roles
from medinet_core.roles.registry import role_registry, RoleDefinition
role_registry.register(RoleDefinition(name='X', permissions={...}))

# Uploaders
from medinet_core.uploaders.registry import uploader_registry
uploader_registry.register(MyUploader)
uploader_cls = uploader_registry.get('.csv')

# Base model
from medinet_core.models import BaseUploadRecord

# CSP nonce in templates
{% load csp_tags %}
<script nonce="{% csp_nonce %}">...</script>

# Permission check
user.has_permission('inference.execute')                  # exists?
user.has_permission('inference.execute', domain='cardio') # scoped?
user.get_permission_scope('inference.execute')            # 'ALL' | [...] | None
```
