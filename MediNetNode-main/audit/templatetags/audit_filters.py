from django import template

register = template.Library()

@register.filter
def lookup(dictionary, key):
    """Template filter to lookup dictionary values by key."""
    if hasattr(dictionary, 'get'):
        return dictionary.get(key, '')
    elif hasattr(dictionary, '__getitem__'):
        try:
            return dictionary[key]
        except (KeyError, IndexError, TypeError):
            return ''
    return ''