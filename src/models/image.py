class ResNetEncoder:
    """Compatibility implementation for the image encoder interface.

    This placeholder accepts the expected constructor arguments and exposes
    the unified ``__call__(image, tabular)`` signature used by model
    factories. Until a full ResNet-backed encoder is implemented, it behaves
    as a no-op encoder and returns the image input unchanged.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, image, tabular):
        return image