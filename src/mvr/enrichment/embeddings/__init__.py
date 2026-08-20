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
from .provider_specs import (
    ProviderSpec,
    ProviderSpecError,
    ProviderSpecRegistry,
    load_provider_specs,
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
    "ProviderSpec",
    "ProviderSpecError",
    "ProviderSpecRegistry",
    "load_provider_specs",
]
