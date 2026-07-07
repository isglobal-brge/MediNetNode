# Bugfix: Rate limiting no aplicaba a peticiones no autenticadas

**Severidad:** Alta
**Sprint:** DP Security & Researcher Budget (2026-04)
**Archivo afectado:** `medinet_core/security/middleware.py`

## Descripción

`RateLimitMiddleware` aplicaba rate limiting solo a peticiones con `request.api_user`
(autenticadas). Las peticiones que fallaban la autenticación (API key incorrecta o ausente)
no tenían ningún límite, permitiendo brute-force ilimitado desde un Hub comprometido.

## Raíz del problema

El check `if self.is_rate_limited(request)` dependía de `api_user` para identificar al usuario.
Las peticiones no autenticadas simplemente pasaban sin limitación.

## Fix

Al inicio de `RateLimitMiddleware.__call__()`, antes de cualquier check autenticado:

```python
_IP_RATE_LIMIT_MAX = 20
_IP_RATE_LIMIT_WINDOW = 60  # segundos

if not hasattr(request, 'api_user'):
    client_ip = self._get_client_ip(request)
    cache_key = f'ratelimit_ip_{client_ip}'
    request_count = cache.get(cache_key, 0)
    if request_count >= _IP_RATE_LIMIT_MAX:
        return JsonResponse({'error': 'Demasiadas peticiones.'}, status=429)
    cache.set(cache_key, request_count + 1, _IP_RATE_LIMIT_WINDOW)
    return self.get_response(request)
```

Usa Django's cache framework (`from django.core.cache import cache`) para tracking por IP.
El contador expira automáticamente en 60 segundos.
