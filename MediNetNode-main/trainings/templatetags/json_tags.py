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
            # If it's already a JSON string, parse and re-format
            data = json.loads(value)
        else:
            # If it's a dict or other object, use it directly
            data = value
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        # If conversion fails, return as string
        return str(value)