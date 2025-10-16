from adapters.mobilesam_adapter.prompt_generator import PromptGenerator
from .mobile_sam_base import MobileSAMBase

class MobileSAMAdapter(MobileSAMBase):
    """
    MobileSAM variant using the SAM-Adapter architecture.
    Only adapter parameters will be trainable.
    """
    def __init__(self, **kwargs):
        # Automatically activate adapter version of the encoder
        super().__init__(**kwargs)
        self.add_adapter(    
            adapter_class=PromptGenerator,
            embed_dim=256,
            scale_factor=4,
            freq_nums=0.25)