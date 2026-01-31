    """
    Gerenciador de configurações que permite sobrescrita.
    """
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        self.config = DEFAULT_CONFIG.copy()
        if config_override:
            self.config.update(config_override)
