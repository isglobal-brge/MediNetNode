"""CSV/Excel formula-injection neutralization.

A spreadsheet interprets a cell whose text begins with '=', '+', '-', '@' (or a
leading control char like tab/CR) as a formula. When exporting attacker-influenced
data (usernames, filenames, audit details), such a cell can execute when an
AUDITOR/ADMIN opens the CSV in Excel. Prefixing a single quote forces literal text.
"""

_DANGEROUS_PREFIXES = ('=', '+', '-', '@', '\t', '\r')


def csv_safe_cell(value):
    """Return a spreadsheet-safe string for a CSV cell."""
    if value is None:
        return ''
    text = str(value)
    if text and text[0] in _DANGEROUS_PREFIXES:
        return "'" + text
    return text
