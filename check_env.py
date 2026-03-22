import torch
import sys
import os

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

try:
    import bitsandbytes
    print(f"bitsandbytes version: {bitsandbytes.__version__}")
except ImportError:
    print("bitsandbytes not installed")
except Exception as e:
    print(f"bitsandbytes error: {e}")

try:
    import peft
    print(f"peft version: {peft.__version__}")
except ImportError:
    print("peft not installed")

try:
    import transformers
    print(f"transformers version: {transformers.__version__}")
except ImportError:
    print("transformers not installed")

try:
    import accelerate
    print(f"accelerate version: {accelerate.__version__}")
except ImportError:
    print("accelerate not installed")
