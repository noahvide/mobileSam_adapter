from .mobile_sam_base import MobileSAMBase

class MobileSAMAdapter(MobileSAMBase):
    """
    MobileSAM variant using the SAM-Adapter architecture.
    Only adapter parameters will be trainable.
    """
    def __init__(self, **kwargs):
        # Automatically activate adapter version of the encoder
        super().__init__(use_adapter=True, **kwargs)