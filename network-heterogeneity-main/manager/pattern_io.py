import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, List
from manager.database.db import DB_ROOT
from classes.pattern import Pattern
from manager.log import dbg, info, warn, err

def pattern_size_to_str(pattern_size: Tuple[int, int]) -> str:
    rows, cols = pattern_size
    size_str = f"{rows}x{cols}"
    return size_str

def list_pattern_indices(
    pattern_id: str,
    pattern_size: Tuple[int, int],
    db_root: Path = DB_ROOT,
) -> List[int]:
    """
    Return a sorted list of integer indices for .bmp patterns in:
        database/patterns/<rows>x<cols>/<pattern_id>/*.bmp
    """
    size_str = pattern_size_to_str(pattern_size)

    pattern_dir = db_root / "patterns" / size_str / pattern_id
    if not pattern_dir.exists():
        err(f"Pattern directory not found: {pattern_dir}")
        raise FileNotFoundError(f"Pattern directory not found: {pattern_dir}")
    
    indices = []
    for bmp in pattern_dir.glob("*.bmp"):
        try:
            indices.append(int(bmp.stem))
        except ValueError:
            # ignore non-numeric filenames
            warn(f"Ignoring non-numeric pattern file: {bmp.name}")
    
    indices.sort()

    if not indices:
        err(f"No .bmp patterns found in {pattern_dir}")
        raise RuntimeError(f"No .bmp patterns found in {pattern_dir}")

    return indices


def count_patterns(pattern_id: str, pattern_size: Tuple[int, int]) -> int:
    return len(list_pattern_indices(pattern_id, pattern_size))


def pattern_from_bitmap(
    path: str,
    identifier: str | None = None,
    normalize: bool = True,
    threshold: float | None = None,
) -> Pattern:
    """
    Load a BMP (or any image) from `path` and convert it to a Pattern.

    Args:
        path: Path to the image file (e.g. "data/patterns/3x3/conv_128x128_circle/0.bmp").
        identifier: Optional identifier string to store in Pattern.identifier.
                    If None, uses the basename of the file.
        normalize: If True, scale pixel values to [0, 1] by dividing by 255.
        threshold: If not None, apply a binary threshold AFTER normalization:
                   values > threshold -> 1.0, else 0.0

    Returns:
        Pattern: pattern whose `array` is a 2D numpy array of shape (H, W).
    """
    path = Path(path)

    if not path.exists():
        err(f"Pattern bitmap not found: {path}")
        raise FileNotFoundError(f"Pattern bitmap not found: {path}")
    
    img = Image.open(path).convert("L")  # "L" = 8-bit grayscale

    arr = np.array(img, dtype=float)

    if normalize:
        arr /= 255.0

    if threshold is not None:
        arr = (arr > threshold).astype(float)

    if identifier is None:
        identifier = os.path.basename(path)

    return Pattern(arr, identifier)