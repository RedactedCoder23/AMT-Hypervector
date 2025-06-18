class LLMAdapter:
    """Tiny adapter placeholder."""

    def __init__(self, model):
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
