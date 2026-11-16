"""FastAPI backend for the SafeSpace check-in demo.

`create_app` is imported lazily so that `safespace.api.support` (which has no web
dependencies) can be used without FastAPI installed.
"""


def create_app(*args, **kwargs):
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app"]
