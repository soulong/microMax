"""Pipeline-specific exceptions (shared by steps and orchestration)."""


class MetadataValidationError(ValueError):
    """Raised when an enabled step requires a metadata column that is absent."""