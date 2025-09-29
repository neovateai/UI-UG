import re
from typing import List, Tuple, Union

def is_english_simple(text: str) -> bool:
    """Check if text contains only ASCII characters."""
    try:
        text.encode(encoding='utf-8').decode('ascii')
        return True
    except UnicodeDecodeError:
        return False

def bbox_2_point(bbox: List[float], precision: int = 2) -> str:
    """Convert bounding box to center point string."""
    if len(bbox) != 4:
        raise ValueError("Bounding box must have exactly 4 coordinates")
    
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return f"({center_x:.{precision}f},{center_y:.{precision}f})"

def bbox_2_bbox(bbox: List[float], precision: int = 2) -> str:
    """Convert bounding box to formatted string."""
    if len(bbox) != 4:
        raise ValueError("Bounding box must have exactly 4 coordinates")
    
    formatted_coords = [f"{coord:.{precision}f}" for coord in bbox]
    return f"({formatted_coords[0]},{formatted_coords[1]},{formatted_coords[2]},{formatted_coords[3]})"

def pred_2_point(s: str) -> List[float]:
    """Extract point coordinates from string representation."""
    floats = [float(num) for num in re.findall(r'-?\d+\.?\d*', s)]
    
    if len(floats) == 2:
        return floats
    elif len(floats) == 4:
        return [(floats[0] + floats[2]) / 2, (floats[1] + floats[3]) / 2]
    else:
        raise ValueError("Invalid point format")

def extract_bbox(s: str) -> List[Tuple[int, int]]:
    """Extract bounding box coordinates from Qwen format strings."""
    patterns = [
        r"<box>\((\d+,\d+)\),\((\d+,\d+)\)</box>",
        r"<\|box_start\|>\((\d+,\s*\d+)\),\((\d+,\s*\d+)\)<\|box_end\|>"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, s)
        if matches:
            # Flatten the matches and convert to integer tuples
            points = []
            for match in matches:
                for coord_str in match:
                    x, y = map(int, coord_str.split(','))
                    points.append((x, y))
            return points
    
    return []