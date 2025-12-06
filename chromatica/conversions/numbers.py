from ..functions import clamp01, RealNumber

class UnitFloat(float):
    """A floating-point number clamped to the inclusive range ``[0, 1]``."""

    def __new__(cls, value: RealNumber):
        if not 0.0 <= value <= 1.0:
            value = clamp01(value)
        return super().__new__(cls, value)

    def __repr__(self):
        return f"UnitFloat({float(self)})"

