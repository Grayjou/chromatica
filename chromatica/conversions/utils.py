from .numbers import UnitFloat

def saturate(value: float) -> UnitFloat:
    """
    Clamp a float value to the [0.0, 1.0] range using UnitFloat.
    """
    return UnitFloat(value)

def hue_to_rgb_components(hue: float) -> tuple[UnitFloat, UnitFloat, UnitFloat]:
    """
    Convert normalized hue (0.0-1.0) to base RGB components before applying saturation/lightness.

    Args:
        hue (float): Hue value normalized to [0.0, 1.0).

    Returns:
        Tuple[UnitFloat, UnitFloat, UnitFloat]: RGB components in [0.0, 1.0].
    """
    r = abs(hue * 6.0 - 3.0) - 1.0
    g = 2.0 - abs(hue * 6.0 - 2.0)
    b = 2.0 - abs(hue * 6.0 - 4.0)
    return saturate(r), saturate(g), saturate(b)