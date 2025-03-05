import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # 根据你的子文件夹深度调整 parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))