import json
from django import template

register = template.Library()


@register.filter
def pretty_json(value):
    """
    Convert dictionary/JSON to pretty formatted JSON string.
    """
    if not value:
        return "No configuration available"
    
    try:
        if isinstance(value, str):
            data = json.loads(value)
        else:
            data = value

        return json.dumps(data, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return str(value)