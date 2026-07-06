"""EDA Engine configuration placeholder."""


class EdaConfigNotImplementedError(NotImplementedError):
    """Raised until Task 31 implements EDA-specific configuration."""


def load_eda_config() -> object:
    raise EdaConfigNotImplementedError("EDA configuration is not implemented yet.")
