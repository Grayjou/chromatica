from ..chromatica.utils import get_dimension

def test_none_dimension():
    assert get_dimension(None) == 0

def test_sized_dimension():
    assert get_dimension([1, 2, 3]) == 3
    assert get_dimension((1, 2)) == 2
    assert get_dimension({"a": 1, "b": 2}) == 2
    assert get_dimension("hello") == 5

def test_non_sized_dimension():
    assert get_dimension(42) == 1
    assert get_dimension(3.14) == 1