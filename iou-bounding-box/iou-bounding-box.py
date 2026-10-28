def iou(box_a: list[float], box_b: list[float]) -> float:
    """
    Returns IoU as a float.
    Each box is given as [x1, y1, x2, y2].
    """
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    # Determine the coordinates of the intersection rectangle
    xL = max(x1_a, x1_b)
    yT = max(y1_a, y1_b)
    xR = min(x2_a, x2_b)
    yB = min(y2_a, y2_b)

    # Calculate intersection area (clamp negative dimensions to zero)
    intersection_width = max(0.0, xR - xL)
    intersection_height = max(0.0, yB - yT)
    intersection_area = intersection_width * intersection_height

    # Calculate individual box areas
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)

    # Calculate union area
    union_area = area_a + area_b - intersection_area

    # Guard against division by zero
    if union_area == 0:
        return 0.0

    return float(intersection_area / union_area)