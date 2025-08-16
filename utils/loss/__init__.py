import sys
from pathlib import Path

ROOT0 = Path(__file__).resolve().parents[0]
if str(ROOT0) not in sys.path:
    sys.path.append(str(ROOT0))

ROOT1 = Path(__file__).resolve().parents[1]
if str(ROOT1) not in sys.path:
    sys.path.append(str(ROOT1))

ROOT2 = Path(__file__).resolve().parents[2]
if str(ROOT2) not in sys.path:
    sys.path.append(str(ROOT2))

print(ROOT2)

from Yolo_V3_loss import *

