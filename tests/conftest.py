import sys
from pathlib import Path

# Add vllm directory to path (same process as in main.py)
vllm_path = str((Path(__file__).parent.parent / "vllm").resolve())
if vllm_path not in sys.path:
    sys.path.insert(0, vllm_path)
