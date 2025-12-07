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


