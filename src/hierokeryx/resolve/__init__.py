"""Resolution layer: within-doc coref and cross-doc clustering."""

from hierokeryx.resolve.coref import resolve_within_doc
from hierokeryx.resolve.crossdoc import build_registry, resolve_crossdoc

__all__ = ["build_registry", "resolve_crossdoc", "resolve_within_doc"]
