
def is_close_to_int(value: float, tol: float = 1e-9) -> bool:
    """Check if a float is close to an integer within a tolerance."""
    return abs(value - round(value)) <= tol