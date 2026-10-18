"""Upstream control: discover, pin, audit, and mirror the code and weights toolkits pull.

Examples:
    >>> from proto_language.provenance.upstream import discover
    >>> lock = discover(Path("proto-tools"))  # doctest: +SKIP
"""

from proto_language.provenance.upstream.discover import discover
from proto_language.provenance.upstream.sources import ToolkitUpstream, UpstreamLock, UpstreamSource

__all__ = ["ToolkitUpstream", "UpstreamLock", "UpstreamSource", "discover"]
