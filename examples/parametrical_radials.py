from typing import List, Callable
from ..chromatica.gradients.full_parametrical_angular_radial import FullParametricalAngularRadialGradient, GradientEnds
from ..chromatica.format_type import FormatType
import numpy as np
from PIL import Image
from numpy.typing import NDArray
DEFAULT_SIZE = (512, 512)

def linear_transform(x: NDArray, min_input, max_input, min_output, max_output) -> NDArray:
    """Linearly transform x from [min_input, max_input] to [min_output, max_output]."""
    # Clip x to input range
    x_clipped = np.clip(x, min_input, max_input)
    # Scale to [0, 1]
    x_scaled = (x_clipped - min_input) / (max_input - min_input)
    # Scale to output range
    return min_output + x_scaled * (max_output - min_output)


def fading_spiral():
    base_radius = 16
    width, height = DEFAULT_SIZE
    offset_deg = -120
    min_angle = 0.0
    min_angle += offset_deg
    max_angle = 330.0
    max_angle += offset_deg
    # Inner ring: full RGB, transparent -> opaque
    # Outer ring: full RGB, transparent -> opaque
    # The color transform will zero out red and alpha, leaving green/blue
    rings: List[GradientEnds] = [
        ((0, 255, 128, 0), (0, 255, 128, 255), (0, 255, 255, 255),  (0, 255, 255, 0)),    # Yellow fading in (will become green after zeroing red)
        ((255, 0, 255, 0), (0, 255, 255, 255), (255, 0, 255, 255), (255, 0, 255, 0)),     # Magenta fading in (will become blue after zeroing red)
        ((255,0, 255,0),(128, 128, 255, 255),(255, 64, 255, 255), (255, 0, 255, 0))
    ]   
    def f(x):
        #x = x*(360 - min_angle)/(max_angle - min_angle)
        #x = (x+offset_deg)%360
        x_radians = np.radians(x)
        return 5 + np.sin(8 * x_radians)
    def h(x):
        #x = x*(360-min_angle)/(max_angle - min_angle)
        #x = (x+offset_deg)%360
        x_radians = np.radians(x)
        unit = x / 360.0
        return 10 + (4*unit + np.sin(2 * x_radians)) * np.sin(x_radians / 2) - 5*(unit**4 + 1-unit**.25) + np.sin(32 * x_radians)/3
    def sine_tight_fade_in_out(x, length=0.1):
        x = np.asarray(x)  # ensure array
        fade_in = (0 <= x) & (x <= length)
        fade_out = (1 - length <= x) & (x <= 1)
        out_of_bounds = (x < 0) | (x > 1)
        return np.where(out_of_bounds, 0, np.where(fade_in | fade_out, np.sin((np.pi/(2*length)) * x), 1))
    def sine_tight_fade_in(x, length=0.1):
        x = np.asarray(x)  # ensure array
        fade_in = (0 <= x) & (x <= length)
        out_of_bounds = (x < 0) | (x > 1)
        return np.where(out_of_bounds, 0, np.where(fade_in, np.sin((np.pi/(2*length)) * x), 1))
    def tight_fade_in_out(x):
        return sine_tight_fade_in_out(x, length=0.1)
    def fade_radius_on_theta_start_end(length=0.1, min_angle=0.0, max_angle=360.0) -> Callable[[NDArray, NDArray], tuple[NDArray, NDArray]]:

        """Fade radius to zero at start and end of theta range."""
       
        def func(radius: NDArray, theta: NDArray) -> tuple[NDArray, NDArray]:
            # Normalize theta to [0, 1] based on the actual angular range
            angle_range = max_angle - min_angle
            unit_theta = (theta - min_angle) / angle_range
            
            # Fade in at the start
            scaled_fade_in = linear_transform(unit_theta, 0.0, length, 0.0, .33)
            scaled_fade_in_c = linear_transform(unit_theta, length, 1.0, .33, 1.0)
            theta_adjusted = np.where(unit_theta < length, scaled_fade_in * angle_range + min_angle,
                                     scaled_fade_in_c * angle_range + min_angle)
            
            # Fade out at the end
            scaled_fade_out = linear_transform(unit_theta, 1.0 - length, 1.0, 1.0-.33, 1.0)
            scaled_fade_out_c = linear_transform(unit_theta, 0.0, 1.0 - length, 0.0, .66)
            theta_adjusted = np.where(unit_theta > 1.0 - length, scaled_fade_out * angle_range + min_angle,
                                     scaled_fade_out_c * angle_range + min_angle)
            
            fade = sine_tight_fade_in(unit_theta, length=length)
            #theta_adjusted = (theta_adjusted + 90.0) % 360.0
            
            return radius*fade, theta_adjusted*359/360
        return func
    def zero_both(r: NDArray, theta: NDArray) -> tuple[NDArray, NDArray]:
        return np.zeros_like(r), np.zeros_like(theta)
    def zero_radius(r: NDArray, theta: NDArray) -> tuple[NDArray, NDArray]:
        return np.zeros_like(r), theta
    def one_radius(r: NDArray, theta: NDArray) -> tuple[NDArray, NDArray]:
        return np.ones_like(r), theta
    def zero_theta(r: NDArray, theta: NDArray) -> tuple[NDArray, NDArray]:
        return r, np.zeros_like(theta)
    gradient = FullParametricalAngularRadialGradient.generate(
        width=width,
        height=height,
        inner_r_theta=lambda theta: base_radius*f(theta),
        outer_r_theta=lambda theta: base_radius*(h(theta)),
        color_rings=rings,
        deg_start=min_angle,
        deg_end=max_angle,
        color_space='rgba',
        format_type=FormatType.INT,
        outside_fill=(0, 0, 0, 0),
        # Use bivariable_color_transforms to nullify channels
        bivariable_space_transforms={
            #0: zero_radius,  # Zero out red channel
            #1: zero_both,  # Zero out red channel
            #2: zero_both,  # Zero out red channel
            #3: fade_radius_on_theta_start_end(0.2, min_angle, max_angle)   # Zero out alpha channel
        },
        rotate_r_theta_with_theta_normalization=True
    )

    img = Image.fromarray(gradient.value.astype(np.uint8), mode='RGBA')
    img.show()


def pad_arc(start_deg: float, end_deg: float, pad_deg: float, clockwise: bool):
    """
    Pads an angular arc by extending it on both ends.

    Clockwise arcs move from start → end in decreasing angle.
    Counterclockwise arcs move from start → end in increasing angle.
    """
    if pad_deg <= 0:
        return start_deg, end_deg
    if clockwise:
        # CW: angles decrease as you move along the arc
        new_start = start_deg + pad_deg  # backward along CW is +pad
        new_end   = end_deg - pad_deg    # forward along CW is -pad
    else:
        # CCW: angles increase as you move along the arc
        new_start = start_deg - pad_deg
        new_end   = end_deg + pad_deg

    return new_start, new_end

WHITE_OUTLINE_WIDTH = 4
WHITE_DEGREE_OUTLINE_WIDTH = 4
def fire_ice_puffy_dagger():
    def outer_r_theta(theta: NDArray) -> NDArray:
        x = np.radians(theta)
        return 3 * (25 + 1.6 * np.sin(1*x) + 
                1.6 * np.sin(20*x + 120*np.pi/180) 
                + 3.2 * np.sin(5*x + 90*np.pi/180) 
                + 3.2 * np.sin(10*x + 90*np.pi/180) 
                + 1.6 * np.sin(4*x + 45*np.pi/180) 
                + 4 * np.sin(6*x + 30*np.pi/180))
    def inner_r_theta(theta: NDArray) -> NDArray:
        x = np.radians(theta)
        return 2*(25 + 2 * np.sin(1*x) 
                + 1 * np.sin(2*x) + 2 * np.sin(3*x) 
                + 5 * np.sin(4*x + 60*np.pi/180) 
                + 1 * np.sin(5*x) + 1 * np.sin(6*x) 
                + 5 * np.sin(7*x + 60*np.pi/180))

    base_radius = 2
    width, height = DEFAULT_SIZE
    offset_deg = 60
    min_angle = 5.0
    min_angle += offset_deg
    max_angle = 280.0
    max_angle += offset_deg
    rings : List[GradientEnds] = [
        ((0, 255, 255),(270, 205, 255),  (180, 180, 230),(200, 157, 255),(160, 80, 255)),
        ((60, 255, 255),(0, 160, 255), (180, 120, 220),(220, 100, 255), (180, 60, 255)),
        ((0, 160, 255),(310, 160, 255), (170, 80, 200),(220, 60, 255), (160, 0, 255)),
    ]
    white_start, white_end = pad_arc(min_angle, max_angle, WHITE_DEGREE_OUTLINE_WIDTH, clockwise=False)
    white_gradient = FullParametricalAngularRadialGradient.generate(
        width=width,
        height=height,
        inner_r_theta=lambda theta: (base_radius*inner_r_theta(theta)*.5) - WHITE_OUTLINE_WIDTH*base_radius*1,
        outer_r_theta=lambda theta: (base_radius*outer_r_theta(theta)) + WHITE_OUTLINE_WIDTH*base_radius,
        color_rings=[((0, 0, 255),(0, 0, 200), (0, 0, 100),(0, 0, 200), (0, 0, 255))],
        deg_start=white_start,
        deg_end=white_end,
        color_space='hsv',
        format_type=FormatType.INT,
        outside_fill=(0, 0, 70),

        #rotate_r_theta_with_theta_normalization=True
    )
    gradient = FullParametricalAngularRadialGradient.generate(
        width=width,
        height=height,
        inner_r_theta=lambda theta: base_radius/2*inner_r_theta(theta),
        outer_r_theta=lambda theta: base_radius*outer_r_theta(theta),
        color_rings=rings,
        deg_start=min_angle,
        deg_end=max_angle,
        color_space='hsv',
        format_type=FormatType.INT,
        outside_fill=white_gradient.value,
        hue_directions_r=["cw",
                           "ccw"
                           ],
        hue_directions_theta=[["ccw", "ccw", "cw", "ccw"],
                               ["ccw", "ccw", "cw", "ccw"], 
                               ["ccw", "ccw", "cw", "ccw"]],
        bivariable_space_transforms={
            #0: lambda r, theta: (r, (theta*(360-min_angle)/(max_angle - min_angle)+offset_deg)%360),
            #0: reverse_theta(0, (360)),
            0: lambda r, theta: (r, (1-theta)**2),
            1: lambda r, theta: (r, (1-theta)**2),
        }
        #rotate_r_theta_with_theta_normalization=True,
    )
    gradient = gradient.convert('rgb', FormatType.INT)
    img = Image.fromarray(gradient.value.astype(np.uint8), mode='RGB')
    img.show()
