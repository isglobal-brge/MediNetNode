from django import template

register = template.Library()


@register.simple_tag(takes_context=True)
def csp_nonce(context):
    """
    Output the CSP nonce for the current request.

    Usage:
        {% load csp_tags %}
        <script nonce="{% csp_nonce %}">...</script>
        <style nonce="{% csp_nonce %}">...</style>

    Returns the nonce string set by SecurityHeadersMiddleware on request.csp_nonce.
    Returns an empty string if no nonce is present (e.g. in tests).
    """
    request = context.get('request')
    nonce = getattr(request, 'csp_nonce', None)
    return nonce or ''
