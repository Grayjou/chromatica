from ...chromatica.gradients.gradient1dv2.gradient_1dv2 import Gradient1D
import numpy as np
from ...chromatica.gradients.gradient1dv2.segment import get_transformed_segment, UniformGradientSegment, TransformedGradientSegment
from ...chromatica.conversions import np_convert
from ...chromatica.gradients.gradient1dv2.interpolator import SequenceMethod
def test_gradient1d_creation():
    grad = Gradient1D.from_colors(
        left_color=(255, 0, 0),
        right_color=(0, 0, 255),
        steps=5,
        color_mode="rgb",
        format_type="int",
    )
    expected_colors = np.array([
        (255, 0, 0),
        (191, 0, 64),
        (128, 0, 128),
        (64, 0, 191),
        (0, 0, 255),
    ])
    assert np.array_equal(grad.value, expected_colors)

def test_gradient1d_with_transforms():
    def transform_channel(unit_x):
        return unit_x ** 2  # Simple quadratic transform
    grad = Gradient1D.from_colors(
        left_color=(0, 0, 0),
        right_color=(255, 255, 255),
        steps=5,
        color_mode="rgb",
        format_type="int",
        per_channel_transforms={0: transform_channel, 1: transform_channel, 2: transform_channel},
    )
    expected_third_color = (64, 64, 64)  # (0.5^2 * 255)
    assert np.array_equal(grad.value[2], expected_third_color)

def test_gradient_sequence():
    grad = Gradient1D.gradient_sequence(
        colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        total_steps=7,
        color_modes=["rgb", "rgb"],
        format_type="int",
    )
    expected_colors = np.array([
        (255, 0, 0),
        (170, 85, 0),
        (85, 170, 0),
        (0, 255, 0),
        (0, 170, 85),
        (0, 85, 170),
        (0, 0, 255),
    ])
    assert np.array_equal(grad.value, expected_colors)

def test_gradient_sequence_with_hue_direction():
    grad = Gradient1D.gradient_sequence(
        colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        total_steps=7,
        color_modes=["hsv", "hsv"],
        format_type="int",
        hue_directions=["longest", "longest"],
        output_color_mode="rgb"
    )
    expected_colors = np.array([
        (255, 0, 0),
        (170, 0, 255),
        (0, 170, 255),
        (0, 255, 0),
        (255, 170, 0),
        (255, 0, 170),
        (0, 0, 255),
    ])

    assert np.array_equal(grad.value, np.array(expected_colors))

    grad = Gradient1D.gradient_sequence(
        colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        total_steps=7,
        color_modes=["rgb", "rgb"],
        format_type="int",
        hue_directions=["shortest", "shortest"],
        output_color_mode="rgb"
    )

    expected_colors_shortest = np.array([
        (255, 0, 0),
        (170, 85, 0),
        (85, 170, 0),
        (0, 255, 0),
        (0, 170, 85),
        (0, 85, 170),
        (0, 0, 255),
    ])
    assert np.array_equal(grad.value, expected_colors_shortest)

def test_gradient_sequence_different_spaces():
    grad = Gradient1D.gradient_sequence(
        colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        total_steps=9,
        input_color_modes=("rgb", "rgb", "rgb"),
        color_modes=["rgb", "hsv"],
        format_type="int",
        output_color_mode="rgb",
    )

    expected_colors = np.array([
        (255, 0, 0),
        (191, 64, 0),
        (128, 128, 0),
        (64, 191, 0),
        (0, 255, 0),
        (0, 255, 128),
        (0, 255, 255),
        (0, 128, 255),
        (0, 0, 255),
    ])

    assert np.array_equal(grad.value, expected_colors)

def test_gradient_sequence_different_lengths():
    grad = Gradient1D.gradient_sequence(
        colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        segment_lengths=[3, 5],
        color_modes=["rgb", "rgb"],
        format_type="int",
        offset=0,
    )

    expected_colors = np.array([
        (255, 0, 0),
        (128, 128, 0),
        (0, 255, 0),
        (0, 191, 64),
        (0, 128, 128),
        (0, 64, 191),
        (0, 0, 255),
    ])

    assert np.array_equal(grad.value, expected_colors)

def test_gradient_sequence_global_transform():
    def global_transform(u):
        return u ** 2  # Quadratic easing

    grad = Gradient1D.gradient_sequence(
        colors=[(0, 0, 0), (0, 0, 255), (255, 0, 0)],
        color_modes=["rgb"],
        format_type="int",
        global_unit_transform=global_transform,
        segment_lengths=[2, 6],
    )

    expected_colors = np.array([
        (0, 0, 0),
        (0, 0, 128),
        (0, 0, 128),
        (0, 0, 184),
        (0, 0, 250),
        (71, 0, 184),
        (158,0, 97),
        (255, 0, 0),
    ])

    assert np.array_equal(grad.value, expected_colors)

def test_single_channel_transform():
    def red_channel_transform(u):
        return u ** 0.5  # Square root easing for red channel

    grad = Gradient1D.from_colors(
        left_color=(0, 0, 0),
        right_color=(255, 255, 255),
        steps=5,
        color_mode="rgb",
        format_type="int",
        per_channel_transforms={0: red_channel_transform},
    )
    expected_colors = np.array([
        (0, 0, 0),
        (128, 64, 64),
        (180, 128, 128),
        (221, 191, 191),
        (255, 255, 255),
    ])
    assert np.array_equal(grad.value, expected_colors)

def test_gradient_sequence_single_channel_transform():
    def green_channel_transform(u):
        return u ** 2  # Quadratic easing for green channel

    grad = Gradient1D.gradient_sequence(
        colors=[(0, 0, 0), (0, 255, 0), (255, 0, 0)],
        total_steps=5,
        color_modes=["rgb", "rgb"],
        format_type="int",
        per_channel_transforms=[{1: green_channel_transform}, None],
    )
    expected_colors = np.array([
        (0, 0, 0),
        (0, 64, 0),
        (0, 255, 0),
        (128, 191, 0),
        (255, 0, 0),
    ])
    assert np.array_equal(grad.value, expected_colors)

def test_gradient_sequence_composite_transform():
    def square_transform(u):
        return u ** 2
    def sqrt_transform(u):
        return u ** 0.5
    grad = Gradient1D.gradient_sequence(
        colors=[(0, 0, 0), (255, 255, 255)],
        total_steps=5,
        color_modes=["rgb"],
        format_type="int",
        per_channel_transforms=[{0: square_transform, 1: square_transform, 2: square_transform}],
        global_unit_transform=sqrt_transform,
    )
    expected_colors = np.array([
        (0, 0, 0),
        (64, 64, 64),
        (127, 127, 127),
        (191, 191, 191),
        (255, 255, 255),
    ])
    grad_no_transform = Gradient1D.gradient_sequence(
        colors=[(0, 0, 0), (255, 255, 255)],
        total_steps=5,
        color_modes=["rgb"],
        format_type="int",
    )
    print(grad.value, "vs", grad_no_transform.value)
    dif = grad.value - grad_no_transform.value
    print("Differences:", dif)
    expected_dif = np.array([
        (0, 0, 0),
        (0, 0, 0),
        (1, 1, 1),
        (0, 0, 0),
        (0, 0, 0),
    ])

    #sqrt(0.5) becomes ~0.7071, squared is 0.49999997, times 255 is ~127.49999, rounded to 127
    # If I round it for a certain epsilon, will it cause unexpected behaviour?
    # Nah, floating point errors are expected in general, abd they are negligibible on non-test sized data
    assert np.array_equal(grad.value, expected_colors)


def test_get_transformed_segment():
    colors = [(255, 0, 0), (0, 255, 0)]
    input_color_modes = ["rgb", "rgb"]
    color_modes = ["hsv"]
    format_type = "int"
    total_steps = 5
    num_segments = 1
    u_scaled = np.linspace(0, 1, total_steps)
    hue_directions = ["shortest"]
    per_channel_transforms = [None]
    bound_type = "clamp"
    colors = [
        np_convert(color, from_space="rgb", to_space="hsv", input_type="int", output_type="float")
        for color in colors
    ]
    print("Converted colors:", colors)
    print("u_scaled:", u_scaled)
    result = get_transformed_segment(
        already_converted_start_color=np.array(colors[0]),
        already_converted_end_color=np.array(colors[-1]),
        per_channel_coords=[u_scaled],
        color_mode=color_modes[0],
        hue_direction=hue_directions[0],

    ).get_value()
    result = np_convert(
        result,
        from_space="hsv",
        to_space="rgb",
        input_type="float",
        output_type="int",
    )
    print("Result",len(result), result)
    expected_colors = np.array([
        (255, 0, 0),
        (191, 64, 0),
        (128, 128, 0),
        (64, 191, 0),
        (0, 255, 0),
    ])
    gradient = Gradient1D.gradient_sequence(
        colors=[(255, 0, 0), (0, 255, 0)],
        total_steps=5,
        color_modes=["hsv"],
        format_type="int",
        output_color_mode="rgb"
    )
    print("Gradient colors:", gradient.value)
    assert np.array_equal(result, gradient.value)



def test_gradient_sequence_segment():
    def identity_global_transform(u):
        return u
    grad = Gradient1D.gradient_sequence(
        colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        total_steps=7,
        color_modes=["rgb", "rgb"],
        format_type="int",
        method=SequenceMethod.SEGMENT,
        global_unit_transform=identity_global_transform
    )
    expected_colors = np.array([
        (255, 0, 0),
        (170, 85, 0),
        (85, 170, 0),
        (0, 255, 0),
        (0, 170, 85),
        (0, 85, 170),
        (0, 0, 255),
    ])
    assert np.array_equal(grad.value, expected_colors)

def test_methods_equivalence():
    def global_transform(u):
        return u ** 2  # Quadratic easing

    grad = Gradient1D.gradient_sequence(
        colors=[(0, 0, 0), (0, 0, 255), (255, 0, 0)],
        color_modes=["rgb"],
        format_type="int",
        global_unit_transform=global_transform,
        segment_lengths=[2, 6],
    )

    expected_colors = np.array([
        (0, 0, 0),
        (0, 0, 128),
        (0, 0, 128),
        (0, 0, 184),
        (0, 0, 250),
        (71, 0, 184),
        (158,0, 97),
        (255, 0, 0),
    ])

    grad_segment = Gradient1D.gradient_sequence(
        colors=[(0, 0, 0), (0, 0, 255), (255, 0, 0)],
        color_modes=["rgb"],
        format_type="int",
        global_unit_transform=global_transform,
        segment_lengths=[2, 6],
        method=SequenceMethod.SEGMENT
    )
    assert np.array_equal(grad.value, expected_colors)
    assert np.array_equal(grad_segment.value, expected_colors)
    grad = Gradient1D.gradient_sequence(
        colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        total_steps=9,
        input_color_modes=("rgb", "rgb", "rgb"),
        color_modes=["rgb", "hsv"],
        format_type="int",
        output_color_mode="rgb",
    )

    expected_colors = np.array([
        (255, 0, 0),
        (191, 64, 0),
        (128, 128, 0),
        (64, 191, 0),
        (0, 255, 0),
        (0, 255, 128),
        (0, 255, 255),
        (0, 128, 255),
        (0, 0, 255),
    ])
    grad_segment = Gradient1D.gradient_sequence(
        colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        total_steps=9,
        input_color_modes=("rgb", "rgb", "rgb"),
        color_modes=["rgb", "hsv"],
        format_type="int",
        output_color_mode="rgb",
        method=SequenceMethod.SEGMENT
    )
    assert np.array_equal(grad.value, expected_colors)
    assert np.array_equal(grad_segment.value, expected_colors)

    def sine_global_transform(u):
        return np.sin((u * np.pi)) # back and forth
    
    grad = Gradient1D.gradient_sequence(
        colors=[(0, 0, 0), (255, 255, 255)],
        total_steps=5,
        color_modes=["rgb"],
        format_type="int",
        per_channel_transforms=[{0: sine_global_transform, 1: sine_global_transform}],
    )

    grad_segment = Gradient1D.gradient_sequence(
        colors=[(0, 0, 0), (255, 255, 255)],
        total_steps=5,
        color_modes=["rgb"],
        format_type="int",
        per_channel_transforms=[{0: sine_global_transform, 1: sine_global_transform}],
        method=SequenceMethod.SEGMENT
    )
    assert np.array_equal(grad.value, grad_segment.value)
