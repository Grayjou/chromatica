from typing import List, Optional, TypeVar, Sequence
from ..utils.list_mismatch import handle_list_size_mismatch
import numpy as np
T = TypeVar("T")

def normalize_height_portions(
    height_portions: Optional[List[float]],
    n_rows: int,
) -> List[float]:
    if n_rows <= 0:
        return []

    if height_portions is None:
        return [1.0 / n_rows] * n_rows

    return normalize_portions(
        [height_portions],
        [n_rows],
        name="height_portions",
    )[0]

def normalize_width_portions(
    width_portions: Optional[List[List[float]]],
    between_color_lengths: List[int],
) -> List[List[float]]:
    if width_portions is None:
        return [
            [1.0 / length] * length if length > 0 else []
            for length in between_color_lengths
        ]

    return normalize_portions(
        width_portions,
        between_color_lengths,
        name="width_portions",
    )


def normalize_2d_rows(
    rows: Optional[List[List[T]]],
    expected_lengths: List[int],
    *,
    default: T,
) -> List[List[T]]:
    """
    Normalize a 2D list row-by-row to expected lengths.

    - If rows is None: fill each row with `default`
    - If row is too short: pad with `default`
    - If row is too long: truncate
    """
    if rows is None:
        return [
            [default] * length
            for length in expected_lengths
        ]

    return [
        handle_list_size_mismatch(row, length, fill_value=default)
        for row, length in zip(rows, expected_lengths)
    ]


def normalize_portions(
    portions: List[List[float]],
    expected_lengths: List[int],
    name: str = "portions",
) -> List[List[float]]:
    normalized: List[List[float]] = []

    for i, row in enumerate(portions):
        expected_len = expected_lengths[i]
        row = list(row)
        n_parts = len(row)
        total = sum(row)

        # -----------------------------
        # Case 1: More portions than colors
        # -----------------------------
        if n_parts > expected_len:
            if total >= 1.0:
                # Keep only matching colors, normalize them
                kept = row[:expected_len]
                kept_sum = sum(kept)

                if kept_sum > 0:
                    kept = [v / kept_sum for v in kept]
                else:
                    kept = [1.0 / expected_len] * expected_len

                row = kept + [0.0] * (n_parts - expected_len)

            else:
                # Distribute remainder over extra portions
                remainder = 1.0 - total
                extras = n_parts - expected_len
                fill = remainder / extras if extras > 0 else 0.0

                row = row[:expected_len] + [
                    v + fill for v in row[expected_len:]
                ]

        # -----------------------------
        # Case 2: Exact match
        # -----------------------------
        elif n_parts == expected_len:
            if not np.isclose(total, 1.0):
                if total <= 0:
                    raise ValueError(
                        f"{name} row {i} sum is {total}, cannot normalize"
                    )
                row = [v / total for v in row]

        # -----------------------------
        # Case 3: Fewer portions than colors
        # -----------------------------
        else:  # n_parts < expected_len
            if total <= 0:
                raise ValueError(
                    f"{name} row {i} sum is {total}, cannot normalize"
                )
            # Truncate colors implicitly by normalizing existing portions
            row = [v / total for v in row]

        normalized.append(row)

    return normalized
