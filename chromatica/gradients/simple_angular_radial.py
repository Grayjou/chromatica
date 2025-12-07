from .gradient import Gradient2D
from typing import Optional, Tuple
from ..colors.types import ColorElement
from ..colors import ColorBase
from ..colors.color import unified_tuple_to_class
from ..normalizers.color_normalizer import normalize_color_input, ColorInput
from ..format_type import FormatType
import numpy as np


def normalize_angle(angle: float) -> float:
    """Normalize angle to [0, 360) range."""
    return angle % 360.0


def interpolate_hue_simple(
    h0: np.ndarray,
    h1: np.ndarray,
    u: np.ndarray,
    direction: Optional[str] = None
) -> np.ndarray:
    """
    Interpolate hue values with wrapping support.
    
    Args:
        h0: Starting hue in degrees (array)
        h1: Ending hue in degrees (array)
        u: Interpolation parameter in [0, 1] (array)
        direction: 'cw' (clockwise), 'ccw' (counter-clockwise), or None (shortest path)
    
    Returns:
        Interpolated hue values
    """
    h0 = h0 % 360.0
    h1 = h1 % 360.0
    
    if direction == 'cw':
        # Always go clockwise (increasing hue)
        mask = h1 <= h0
        h1 = np.where(mask, h1 + 360.0, h1)
    elif direction == 'ccw':
        # Always go counter-clockwise (decreasing hue)
        mask = h1 >= h0
        h1 = np.where(mask, h1 - 360.0, h1)
    else:
        # Shortest path
        delta = h1 - h0
        h1 = np.where(delta > 180.0, h1 - 360.0, h1)
        h1 = np.where(delta < -180.0, h1 + 360.0, h1)
    
    dh = h1 - h0
    return (h0 + u * dh) % 360.0

def validate_and_return_outside_fill_array(arr: np.ndarray, width:int, height:int, num_channels:int) -> np.ndarray | Tuple:
    if arr.ndim == 1:
        return tuple(arr.tolist())
    else:
        #check the if the shape matches
        expected_shape = (height, width, num_channels)
        if arr.shape != expected_shape:
            raise ValueError("outside_fill array shape does not match the expected image shape.")
        return arr
    
def process_outside_fill(outside_fill: Optional[ColorInput], width:int, height:int, format_type: FormatType, color_space: str) -> np.ndarray:
    respective_class = unified_tuple_to_class[(color_space, format_type)]
    num_channels = respective_class.num_channels
    #null_value = respective_class.null_value
    if outside_fill is None:
        # Return an array of zeros with shape (height, width, channels)
        return np.zeros((height, width, num_channels))
    if isinstance(outside_fill, np.ndarray):
        return validate_and_return_outside_fill_array(outside_fill, width, height, num_channels)
    elif isinstance(outside_fill, ColorBase):
        if isinstance(outside_fill.value, np.ndarray):
            return validate_and_return_outside_fill_array(outside_fill.value, width, height, num_channels)
        else:
            value = normalize_color_input(outside_fill)
            # return an array filled with this value of shape (height, width, channels)
            return np.full((height, width, num_channels), value)
    else:
        value = normalize_color_input(outside_fill)
        # Ensure it's always an array
        return np.array(value) if not isinstance(value, np.ndarray) else value

class SimpleAngularRadialGradient(Gradient2D):
    @classmethod
    def generate(cls,
                width: int,
                height: int,
                radius: float,
                inner_ring_colors: Tuple[ColorInput, ColorInput],
                outer_ring_colors: Tuple[ColorInput, ColorInput],
                color_space: str = 'rgb',
                format_type: FormatType = FormatType.FLOAT,
                center: Optional[Tuple[int, int]] = None,
                relative_center: Optional[Tuple[float, float]] = None,
                deg_start:float = 0.0,
                deg_end: float = 360.0,
                radius_start: float = 0.0,
                radius_end: float = 1.0,
                normalize_radius: bool = True, # Normalize between radius start and end
                normalize_theta: bool = True, # Normalize between deg start and end
                hue_direction_theta: Optional[str] = None,  # 'cw', 'ccw', or None for angular interpolation
                hue_direction_r: Optional[str] = None,      # 'cw', 'ccw', or None for radial interpolation
                outside_fill: Optional[ColorInput] = None) -> 'Gradient2D':
        
        start_color1 = normalize_color_input(inner_ring_colors[0])
        start_color2 = normalize_color_input(inner_ring_colors[1])
        end_color1 = normalize_color_input(outer_ring_colors[0])
        end_color2 = normalize_color_input(outer_ring_colors[1])
    
        if center is None:
            if relative_center is not None:
                center_x = int(relative_center[0] * width)
                center_y = int(relative_center[1] * height)
                center = (center_x, center_y)
            else:
                center = (width // 2, height // 2)
        
        outside_fill_processed = process_outside_fill(outside_fill, width, height, format_type, color_space)
        
        # Generate indices np matrix
        indices_matrix = np.indices((height, width), dtype=np.float32)
        
        # Calculate radius and degrees matrices
        y_indices = indices_matrix[0] - center[1]
        x_indices = indices_matrix[1] - center[0]
        
        # Calculate distance from center
        distances = np.sqrt(x_indices**2 + y_indices**2)
        
        # Calculate angles in degrees
        theta = (np.degrees(np.arctan2(y_indices, x_indices)) + 360.0) % 360.0
        
        # Create angular mask - handle negative angles and ranges > 360 degrees
        angular_mask = np.ones_like(theta, dtype=bool)
        if normalize_theta:
            # Calculate the effective angular range
            if deg_end >= deg_start:
                angular_range = deg_end - deg_start
                if angular_range >= 360.0:
                    # Full circle or more - include everything
                    angular_mask = np.ones_like(theta, dtype=bool)
                else:
                    # Normal case
                    theta_start = deg_start % 360.0
                    theta_end = deg_end % 360.0
                    if theta_end >= theta_start:
                        angular_mask = (theta >= theta_start) & (theta <= theta_end)
                    else:
                        # Wrap-around at 360
                        angular_mask = (theta >= theta_start) | (theta <= theta_end)
            else:
                # Wrap-around case (deg_end < deg_start)
                theta_start = deg_start % 360.0
                theta_end = deg_end % 360.0
                # Normal wrap-around
                angular_mask = (theta >= theta_start) | (theta <= theta_end)
        
        # Normalize theta to [0, 1] based on the angular range
        if normalize_theta:
            # Calculate the effective angular range (accounting for negative start angles)
            # We need to map [deg_start, deg_end] to [0, 1]
            if deg_end >= deg_start:
                theta_range = deg_end - deg_start
                if theta_range <= 0:
                    theta_range = 360.0  # Full circle
            else:
                # Wrap-around case
                theta_range = (360.0 - normalize_angle(deg_start)) + normalize_angle(deg_end)
            
            # Adjust theta relative to deg_start
            theta_adjusted = (theta - deg_start + 360.0) % 360.0
            
            # Handle wrap-around case
            if deg_end < deg_start:
                # Wrap-around: e.g., deg_start=270, deg_end=90
                theta_normalized = theta_adjusted / theta_range
                # For wrap-around, values > 1 should wrap back to [0, 1]
                theta_normalized = theta_normalized % 1.0
            else:
                # Normal case - clip to [0, 1] within the angular range
                theta_normalized = np.where(angular_mask, 
                                          np.clip(theta_adjusted / theta_range, 0.0, 1.0), 
                                          0)
        else:
            theta_normalized = theta / 360.0
        
        # Calculate r_min and r_max based on radius_start and radius_end
        r_min = radius * radius_start
        r_max = radius * radius_end
        
        # Normalize radius to [0, 1] based on radius_start and radius_end
        if normalize_radius:
            # Map from [r_min, r_max] to [0, 1]
            denominator = r_max - r_min
            denominator = np.where(np.abs(denominator) < 1e-3, 1e-3, denominator)
            u_r = (distances - r_min) / denominator
        else:
            # When normalize_radius=False, still interpolate from r_min to r_max
            # but use actual distance-based calculation
            denominator = r_max - r_min
            denominator = np.where(np.abs(denominator) < 1e-3, 1e-3, denominator)
            u_r = (distances - r_min) / denominator
        
        # Create radial mask for pixels within the radial bounds
        radial_mask = (distances >= r_min) & (distances <= r_max)
        
        # Create combined mask for pixels within both angular and radial bounds
        if normalize_theta:
            combined_mask = angular_mask & radial_mask
        else:
            combined_mask = radial_mask
        
        # Clip u_r to [0, 1] for gradient calculation
        u_r = np.clip(u_r, 0.0, 1.0)
        
        # Get number of channels
        num_channels = len(start_color1)
        result = np.zeros((height, width, num_channels), dtype=np.float32)
        
        # Check if we're in a hue-based color space
        is_hue_space = color_space.lower() in ('hsv', 'hsl', 'hsva', 'hsla')
        
        # Linear interpolation in angular dimension
        # inner_ring_colors[0] -> inner_ring_colors[1] as theta goes from deg_start to deg_end
        # outer_ring_colors[0] -> outer_ring_colors[1] as theta goes from deg_start to deg_end
        for ch in range(num_channels):
            # Interpolate along theta for inner radius (inner_ring_colors)
            if ch == 0 and is_hue_space:
                # ALWAYS use hue interpolation for the hue channel (channel 0) in HSV/HSL spaces
                # Use shortest path if no direction specified (None)
                inner_color = interpolate_hue_simple(
                    np.full_like(theta_normalized, start_color1[ch]),
                    np.full_like(theta_normalized, start_color2[ch]),
                    theta_normalized,
                    hue_direction_theta  # Can be None for shortest path
                )
            else:
                # Linear interpolation for other channels
                inner_color = start_color1[ch] * (1 - theta_normalized) + start_color2[ch] * theta_normalized
            
            # Interpolate along theta for outer radius (outer_ring_colors)
            if ch == 0 and is_hue_space:
                # ALWAYS use hue interpolation for the hue channel
                outer_color = interpolate_hue_simple(
                    np.full_like(theta_normalized, end_color1[ch]),
                    np.full_like(theta_normalized, end_color2[ch]),
                    theta_normalized,
                    hue_direction_theta
                )
            else:
                # Linear interpolation for other channels
                outer_color = end_color1[ch] * (1 - theta_normalized) + end_color2[ch] * theta_normalized
            
            # Interpolate along radius
            if ch == 0 and is_hue_space:
                # ALWAYS use hue interpolation for radial dimension
                channel_gradient = interpolate_hue_simple(
                    inner_color,
                    outer_color,
                    u_r,
                    hue_direction_r
                )
            else:
                # Linear interpolation for radial dimension
                channel_gradient = inner_color * (1 - u_r) + outer_color * u_r
            
            # Apply gradient only to pixels within the combined mask
            result[..., ch] = np.where(combined_mask, channel_gradient, 0)
        
        # Apply outside fill to areas beyond the gradient
        outside_areas = ~combined_mask
        if isinstance(outside_fill_processed, np.ndarray):
            if outside_fill_processed.ndim == 1:
                # Single color tuple - apply to all outside areas
                for ch in range(num_channels):
                    result[outside_areas, ch] = outside_fill_processed[ch]
            else:
                # Full image array - apply to masked areas
                result[outside_areas] = outside_fill_processed[outside_areas]
        
        # Convert to appropriate format
        if format_type == FormatType.INT:
            result = np.round(result).astype(np.uint16)
        
        # Create ColorBase instance and wrap in Gradient2D
        assigned_class = unified_tuple_to_class[(color_space, format_type)]
        result = assigned_class(result)
        # Create Gradient2D
        gradient_obj = cls(result)
        
        return gradient_obj

def example_simple_angular_radial():
    """Example: Simple angular-radial gradient with linear interpolation."""
    from PIL import Image
    
    gradient = SimpleAngularRadialGradient.generate(
        width=500,
        height=500,
        radius=150,
        inner_ring_colors=(
            (255, 0, 0, 0),      # Red at center, deg_start
            (0, 255, 0, 0)       # Green at center, deg_end
        ),
        outer_ring_colors=(
            (0, 0, 255, 255),      # Blue at edge, deg_start
            (255, 255, 0, 255)     # Yellow at edge, deg_end
        ),
        deg_start=0.0,
        deg_end=300.0,
        radius_start=0.5,
        radius_end=1.0,
        color_space='rgba',
        format_type=FormatType.INT,
        outside_fill=(0, 0, 0, 0)
    )
    
    img = Image.fromarray(gradient.value.astype(np.uint8), mode='RGBA')
    img.show()


def example_simple_angular_radial_quadrant():
    """Example: Simple angular-radial gradient covering only one quadrant."""
    from PIL import Image
    
    gradient = SimpleAngularRadialGradient.generate(
        width=500,
        height=500,
        radius=50,
        inner_ring_colors=(
            (255, 0, 0),      # Red at center, 0°
            (0, 255, 0)       # Green at center, 90°
        ),
        outer_ring_colors=(
            (0, 0, 255),      # Blue at edge, 0°
            (255, 255, 0)     # Yellow at edge, 90°
        ),
        deg_start=0.0,
        deg_end=90.0,
        radius_start=0.0,
        radius_end=1.0,
        color_space='rgb',
        format_type=FormatType.INT,
        outside_fill=(50, 50, 50)
    )
    
    img = Image.fromarray(gradient.value.astype(np.uint8), mode='RGB')
    img.show()


if __name__ == '__main__':
    print("Generating simple angular-radial gradient examples...")
    example_simple_angular_radial()
    example_simple_angular_radial_quadrant()


def example_hsv_hue_directions():
    """Example: HSV gradient with hue direction control."""
    from PIL import Image
    
    # Create gradient with clockwise hue interpolation
    gradient = SimpleAngularRadialGradient.generate(
        width=500,
        height=500,
        radius=200,
        inner_ring_colors=(
            (0, 100, 100),      # Red hue at center, deg_start
            (120, 100, 100)     # Green hue at center, deg_end
        ),
        outer_ring_colors=(
            (0, 100, 50),       # Red hue at edge, deg_start
            (120, 100, 50)      # Green hue at edge, deg_end
        ),
        deg_start=0.0,
        deg_end=360.0,
        radius_start=0.3,
        radius_end=1.0,
        color_space='hsv',
        format_type=FormatType.INT,
        hue_direction_theta='cw',  # Clockwise hue interpolation angularly
        hue_direction_r='ccw',     # Counter-clockwise hue interpolation radially
        outside_fill=(0, 0, 0)
    )
    
    img = Image.fromarray(gradient.value.astype(np.uint8), mode='HSV').convert('RGB')
    img.show()
    
    print("HSV gradient with hue directions - clockwise angular, counter-clockwise radial")
