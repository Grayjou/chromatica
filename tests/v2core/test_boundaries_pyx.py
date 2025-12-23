from ...chromatica.v2core.border_handler import (
    handle_border_edges_2d,
    handle_border_lines_2d, 
    BORDER_REPEAT,
    BORDER_MIRROR,
    BORDER_CONSTANT,
    BORDER_CLAMP,
    BORDER_OVERFLOW,
)

def test_import_border_handling():
    assert handle_border_edges_2d is not None
    x_values = [-2, -1, 0, .99, 2]
    x_repeat = [0, 0, 0, .99, 0]
    x_mirror = [0, 1, 0, .99, 0]
    x_clamp = [0, 0, 0, .99, 1]
    x_constant = [None, None, 0, .99, None]
    for i, x in enumerate(x_values):
        assert handle_border_edges_2d(x, x, BORDER_REPEAT)[0] == x_repeat[i]
        assert handle_border_edges_2d(x, x, BORDER_REPEAT)[1] == x_repeat[i]
        assert handle_border_edges_2d(x, x, BORDER_MIRROR)[0] == x_mirror[i]  
        assert handle_border_edges_2d(x, x, BORDER_MIRROR)[1] == x_mirror[i]
        assert handle_border_edges_2d(x, x, BORDER_CLAMP)[0] == x_clamp[i]
        assert handle_border_edges_2d(x, x, BORDER_CLAMP)[1] == x_clamp[i]
        assert handle_border_edges_2d(x, x, BORDER_OVERFLOW)[0] == x
        assert handle_border_edges_2d(x, x, BORDER_OVERFLOW)[1] == x
        if x_constant[i] is None:
            assert handle_border_edges_2d(x, x, BORDER_CONSTANT) is None
        else:
            assert handle_border_edges_2d(x, x, BORDER_CONSTANT)[0] == x_constant[i]

def test_import_border_lines():
    assert handle_border_lines_2d is not None
    y_values = [-2, -1, 0, .99, 2]
    y_repeat = [0, 0, 0, .99, 0]
    y_mirror = [0, 1, 0, .99, 0]
    y_clamp = [0, 0, 0, .99, 1]
    y_constant = [None, None, 0, .99, None]
    for i, y in enumerate(y_values):
        assert handle_border_lines_2d(y, y, BORDER_REPEAT)[1] == y_repeat[i]
        assert handle_border_lines_2d(y, y, BORDER_MIRROR)[1] == y_mirror[i]  
        assert handle_border_lines_2d(y, y, BORDER_CLAMP)[1] == y_clamp[i]
        assert handle_border_lines_2d(y, y, BORDER_OVERFLOW)[1] == y
        assert handle_border_lines_2d(y, y, BORDER_OVERFLOW)[0] == y_clamp[i] #clamps line axis
        if y_constant[i] is None:
            assert handle_border_lines_2d(y, y, BORDER_CONSTANT) is None
        else:
            assert handle_border_lines_2d(y, y, BORDER_CONSTANT)[1] == y_constant[i]