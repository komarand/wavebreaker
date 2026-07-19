from __future__ import annotations


class SourceRegistryError(RuntimeError):
    """Base error for persistent source-registry operations."""


class SourceIdentityError(SourceRegistryError):
    """Raised when a stable source identity cannot be derived."""


class SourceCacheMissError(SourceRegistryError):
    """Raised when a required cache entry is unavailable."""


class SourceOfflineCacheMissError(SourceCacheMissError):
    """Raised when offline mode cannot satisfy a request from cached data."""


class SourceVersionError(SourceRegistryError):
    """Raised for invalid or corrupt source-version state."""


class ArtifactCacheError(SourceRegistryError):
    """Raised for invalid or corrupt artifact-cache state."""


class EmbeddingCompatibilityError(SourceRegistryError):
    """Raised when an embedding is incompatible with configured storage."""


class SourceMigrationError(SourceRegistryError):
    """Raised when a source-registry migration cannot complete safely."""
