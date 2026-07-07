"""
Endpoints para gestión del presupuesto epsilon por researcher.

- POST /api/v2/budget-reset/                     — researcher solicita reset
- POST /api/v2/budget-reset/<id>/approve/        — admin aprueba
- POST /api/v2/budget-reset/<id>/reject/         — admin rechaza
- GET  /api/v2/budget-reset/                     — researcher/admin lista solicitudes
"""
import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import get_object_or_404

from trainings.models import BudgetResetRequest
from dataset.models import ResearcherEpsilonBudget

logger = logging.getLogger(__name__)


def _is_admin(user) -> bool:
    return bool(user and user.role and user.role.name == 'ADMIN')


def _is_researcher(user) -> bool:
    return bool(user and user.role and user.role.name == 'RESEARCHER')


@csrf_exempt
@require_http_methods(["POST", "GET"])
def request_budget_reset(request):
    """POST: researcher solicita reset de su presupuesto en un dataset."""
    user = getattr(request, 'api_user', None) or getattr(request, 'user', None)
    if not _is_researcher(user):
        return JsonResponse({'error': 'Solo los researchers pueden solicitar un reset.'}, status=403)

    if request.method == 'GET':
        requests = BudgetResetRequest.objects.filter(researcher_id=user.id).order_by('-requested_at')
        data = [
            {
                'id': r.id,
                'dataset_id': r.dataset_id,
                'status': r.status,
                'reason': r.reason,
                'requested_at': r.requested_at.isoformat(),
                'review_notes': r.review_notes,
            }
            for r in requests
        ]
        return JsonResponse({'results': data})

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'JSON inválido.'}, status=400)

    dataset_id = body.get('dataset_id')
    reason = (body.get('reason') or '').strip()

    if not dataset_id or not isinstance(dataset_id, int):
        return JsonResponse({'error': 'dataset_id es obligatorio y debe ser un entero.'}, status=400)
    if not reason:
        return JsonResponse({'error': 'El motivo de la solicitud es obligatorio.'}, status=400)
    if len(reason) > 1000:
        return JsonResponse({'error': 'El motivo no puede superar 1000 caracteres.'}, status=400)

    if BudgetResetRequest.objects.filter(
        dataset_id=dataset_id, researcher_id=user.id, status='pending'
    ).exists():
        return JsonResponse(
            {'error': 'Ya tienes una solicitud pendiente para este dataset.'}, status=409
        )

    reset_req = BudgetResetRequest.objects.create(
        dataset_id=dataset_id,
        researcher_id=user.id,
        reason=reason,
    )
    return JsonResponse({'id': reset_req.id, 'status': 'pending'}, status=201)


@csrf_exempt
@require_http_methods(["POST"])
def approve_budget_reset(request, request_id):
    """POST: admin aprueba una solicitud de reset y aplica el reset."""
    user = getattr(request, 'api_user', None) or getattr(request, 'user', None)
    if not _is_admin(user):
        return JsonResponse({'error': 'Solo los administradores pueden aprobar solicitudes.'}, status=403)

    reset_req = get_object_or_404(BudgetResetRequest, pk=request_id)

    # BudgetResetRequest (default db) and ResearcherEpsilonBudget (datasets_db)
    # live in different databases, so a single cross-DB transaction is not
    # possible. Instead we ORDER the operations so the harmful partial state
    # ("approved but reset never applied") can never occur: validate, apply the
    # reset, and only then finalize the approval. The reset is effectively
    # idempotent (it zeroes spent_epsilon), so a crash between reset and approve
    # leaves the request re-approvable with no inconsistency.
    if reset_req.status != 'pending':
        return JsonResponse({'error': 'Esta solicitud ya ha sido revisada.'}, status=409)

    try:
        body = json.loads(request.body) if request.body else {}
    except (json.JSONDecodeError, ValueError):
        body = {}

    notes = (body.get('notes') or '').strip()

    # 1. Apply the researcher budget reset first.
    try:
        budget = ResearcherEpsilonBudget.objects.get(
            dataset_id=reset_req.dataset_id,
            researcher_id=reset_req.researcher_id,
        )
    except ResearcherEpsilonBudget.DoesNotExist:
        budget = None
        logger.warning(
            "Reset request has no ResearcherEpsilonBudget yet: "
            "researcher_id=%s dataset_id=%s — approving without a reset to apply.",
            reset_req.researcher_id, reset_req.dataset_id,
        )

    if budget is not None:
        try:
            budget.reset_period()
        except Exception as exc:  # do NOT approve if the reset could not be applied
            logger.error(
                "Failed to apply budget reset (researcher_id=%s dataset_id=%s): %s — "
                "leaving request pending.",
                reset_req.researcher_id, reset_req.dataset_id, exc,
            )
            return JsonResponse(
                {'error': 'No se pudo aplicar el reset del presupuesto. Solicitud sin cambios.'},
                status=500,
            )

    # 2. Finalize the approval only after the reset succeeded.
    try:
        reset_req.approve(admin=user, notes=notes)
    except ValueError as exc:
        return JsonResponse({'error': str(exc)}, status=409)

    logger.info(
        "Budget reset by admin %s: researcher_id=%s dataset_id=%s",
        user.username, reset_req.researcher_id, reset_req.dataset_id,
    )
    return JsonResponse({'id': reset_req.id, 'status': 'approved'})


@csrf_exempt
@require_http_methods(["POST"])
def reject_budget_reset(request, request_id):
    """POST: admin rechaza una solicitud de reset."""
    user = getattr(request, 'api_user', None) or getattr(request, 'user', None)
    if not _is_admin(user):
        return JsonResponse({'error': 'Solo los administradores pueden rechazar solicitudes.'}, status=403)

    reset_req = get_object_or_404(BudgetResetRequest, pk=request_id)

    try:
        body = json.loads(request.body) if request.body else {}
    except (json.JSONDecodeError, ValueError):
        body = {}

    notes = (body.get('notes') or '').strip()

    try:
        reset_req.reject(admin=user, notes=notes)
    except ValueError as exc:
        return JsonResponse({'error': str(exc)}, status=409)

    return JsonResponse({'id': reset_req.id, 'status': 'rejected'})
