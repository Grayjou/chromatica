from ...chromatica.gradients.gradientv1.gradient2d import Gradient2D, Color1DArr
from ...chromatica.colors.rgb import RGB
import numpy as np
def test_gradient2d_initialization():
    # Test initialization of Gradient2D with default parameters
    top = [(255, 0, 0)] * 100 + [(0, 0, 255)] * 100 + [(255, 0, 0)]*100
    bottom = [(1, 0, 0)]*100 + [(0, 255, 0)]*100 + [(0, 255, 255)]*100

    red = RGB(np.array([(255, 0, 0), (255, 0, 0), (255, 0, 0)]))
    red_hsv = red.convert('hsv', 'int')

    top = np.array(top)


    bottom = np.array(bottom)

    top = RGB(top)

    bottom = RGB(bottom)
    top = top.convert('hsv', 'int')

    bottom = bottom.convert('hsv', 'int')

    gradient = Gradient2D.from_1d_arrays(top, bottom, format_type='int', height=200, color_space='hsv')
    from PIL import Image
    gradient = gradient.convert('rgb', 'int')
    arr = gradient.value.astype('uint8')
    img = Image.fromarray(arr, 'RGB')

    