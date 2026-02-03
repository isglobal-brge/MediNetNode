"""
Custom error handlers for secure error responses.

Prevents information disclosure in production while maintaining debugging in development.
"""
import logging
from django.http import JsonResponse
from django.conf import settings

logger = logging.getLogger(__name__)


def handler400(request, exception):
    """Handle 400 Bad Request errors."""
    logger.warning(f"400 Bad Request: {request.path} - {str(exception)}")

    return JsonResponse({
        'error': 'Bad Request',
        'status': 400
    }, status=400)


def handler403(request, exception):
    """Handle 403 Forbidden errors."""
    logger.warning(f"403 Forbidden: {request.path} from IP {request.META.get('REMOTE_ADDR')}")

    return JsonResponse({
        'error': 'Forbidden',
        'status': 403
    }, status=403)


def handler404(request, exception):
    """Handle 404 Not Found errors."""
    logger.info(f"404 Not Found: {request.path}")

    return JsonResponse({
        'error': 'Not Found',
        'status': 404
    }, status=404)


def handler500(request):
    """Handle 500 Internal Server Error."""
    logger.error(f"500 Internal Server Error: {request.path} from IP {request.META.get('REMOTE_ADDR')}")

    # Generic error message in production
    return JsonResponse({
        'error': 'Internal Server Error',
        'status': 500
    }, status=500)


class SafeErrorResponse:
    """
    Helper class for safe error responses in API views.

    Usage:
        try:
            # risky operation
        except Exception as e:
            return SafeErrorResponse.internal_error(request, e, "operation description")
    """

    @staticmethod
    def internal_error(request, exception, operation=None):
        """
        Return safe internal error response.

        Args:
            request: Django request object
            exception: The caught exception
            operation: Optional description of operation that failed

        Returns:
            JsonResponse with generic error message
        """
        # Log full error details for debugging
        error_details = {
            'path': request.path,
            'method': request.method,
            'ip': request.META.get('REMOTE_ADDR'),
            'user': getattr(request, 'api_user', None) or getattr(request, 'user', None),
            'exception': str(exception),
            'type': type(exception).__name__
        }

        if operation:
            error_details['operation'] = operation

        logger.error(
            f"Internal error in {operation or request.path}: "
            f"{type(exception).__name__}: {str(exception)}",
            extra=error_details
        )

        # Return generic message (no sensitive details)
        return JsonResponse({
            'error': 'Internal server error',
            'status': 500
        }, status=500)

    @staticmethod
    def validation_error(message, status=400):
        """
        Return validation error response.

        Args:
            message: User-friendly error message
            status: HTTP status code (default: 400)

        Returns:
            JsonResponse with validation error
        """
        return JsonResponse({
            'error': message,
            'status': status
        }, status=status)

    @staticmethod
    def permission_denied(message="Permission denied"):
        """
        Return permission denied response.

        Args:
            message: User-friendly error message

        Returns:
            JsonResponse with 403 status
        """
        return JsonResponse({
            'error': message,
            'status': 403
        }, status=403)

    @staticmethod
    def not_found(message="Resource not found"):
        """
        Return not found response.

        Args:
            message: User-friendly error message

        Returns:
            JsonResponse with 404 status
        """
        return JsonResponse({
            'error': message,
            'status': 404
        }, status=404)

    @staticmethod
    def unauthorized(message="Unauthorized"):
        """
        Return unauthorized response.

        Args:
            message: User-friendly error message

        Returns:
            JsonResponse with 401 status
        """
        return JsonResponse({
            'error': message,
            'status': 401
        }, status=401)
