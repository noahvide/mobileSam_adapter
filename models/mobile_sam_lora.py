from .mobile_sam_base import MobileSAMBase

class MobileSAMLoRA(MobileSAMBase):
    """
    MobileSAM variant with LoRA fine-tuning.
    Only LoRA layers will be trainable.
    """
    def __init__(self, rank=4, alpha=16, **kwargs):
        # Pass LoRA parameters through kwargs for flexibility
        super().__init__(**kwargs)
        self.rank = rank
        self.alpha = alpha
        self.add_lora(r=self.rank, alpha=self.alpha)