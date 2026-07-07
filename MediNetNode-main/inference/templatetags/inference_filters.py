"""
Template filters for inference app.
"""
from django import template

register = template.Library()


@register.filter
def get_item(dictionary, key):
    """
    Get an item from a dictionary using a variable key.

    Usage: {{ my_dict|get_item:key_variable }}

    Example:
        {% with pred.label|stringformat:"s" as label_key %}
            {{ class_labels|get_item:label_key }}
        {% endwith %}
    """
    if dictionary is None:
        return None
    return dictionary.get(str(key))


@register.filter
def get_class_label(class_labels, label):
    """
    Get class label name from class_labels dict.
    Falls back to 'Class X' if not found.

    Usage: {{ class_labels|get_class_label:pred.label }}
    """
    if class_labels is None or label is None:
        return f"Class {label}"

    label_str = str(label)
    if label_str in class_labels:
        return class_labels[label_str]

    # Try integer key
    if label in class_labels:
        return class_labels[label]

    return f"Class {label}"
