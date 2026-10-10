"""Original ITRA model plus the standalone reproduction package.

Load the legacy model lazily so independent evaluation and tests do not require
optional ChineseCLIP, adapter, and sentence-transformer packages.
"""


def get_model(*args, **kwargs):
    from model.model import get_model as original_get_model
    return original_get_model(*args, **kwargs)

