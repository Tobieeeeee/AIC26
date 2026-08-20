"""Image/text embedding providers and their offline artifact contracts."""

from .artifacts import (
    EmbeddingArtifactError,
    EmbeddingManifest,
    EmbeddingMappingRecord,
    EmbeddingProviderArtifact,
    EmbeddingProviderRegistry,
    discover_embedding_providers,
    load_mapping,
    load_provider_artifact,
    sha256_file,
    write_manifest,
)

__all__ = [
    "EmbeddingArtifactError",
    "EmbeddingManifest",
    "EmbeddingMappingRecord",
    "EmbeddingProviderArtifact",
    "EmbeddingProviderRegistry",
    "discover_embedding_providers",
    "load_mapping",
    "load_provider_artifact",
    "sha256_file",
    "write_manifest",
]
