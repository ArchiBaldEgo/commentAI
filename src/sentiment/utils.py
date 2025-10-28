import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict

import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def save_json(data: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
