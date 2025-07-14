# tests/conftest.py
import sys
import os
from pathlib import Path

# Add src directory to Python path for imports
# This runs automatically for all tests in this directory
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Shared fixtures for all tests
import pytest
import torch
from torch import nn

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional
import tempfile
import os


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed

