"""Per-request / per-operation context (workspace, user, access scope).

The global :class:`telesearch.config.Settings` covers process-wide model and
device configuration. ``RequestContext`` carries the *per-operation* facts that
multi-tenancy needs: which workspace's data we touch, who is asking, and which
collections the search may read. The CLI uses an implicit default context, while
an API layer constructs one per authenticated request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

DEFAULT_WORKSPACE = "default"


@dataclass
class RequestContext:
    """Scope for an indexing or search operation."""

    workspace_id: str = DEFAULT_WORKSPACE
    user_id: Optional[str] = None
    role: str = "owner"
    # Collections the operation may read. ``None`` means "all collections in
    # this workspace" (the single-user CLI case).
    collections: Optional[list[str]] = field(default=None)

    @classmethod
    def default(cls) -> "RequestContext":
        return cls()
