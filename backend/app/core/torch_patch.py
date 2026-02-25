
import torch
import logging

logger = logging.getLogger(__name__)

# Monkey Patch torch.int1-int8 if missing
# This fixes the AttributeError when importing transformers with incompatible torchao
# Issue: torchao expects torch.int1-8, but current torch version might lack them.
def apply_torch_patch():
    patched_count = 0
    for i in range(1, 9):
        attr = f"int{i}"
        if not hasattr(torch, attr):
            # potentially log debug info
            setattr(torch, attr, torch.int8)
            patched_count += 1
    
    if patched_count > 0:
        logger.warning(f"🔧 [System] Patched torch with {patched_count} missing int types (int1-int8) for compatibility.")

apply_torch_patch()
