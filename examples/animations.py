from PIL import Image, ImageDraw
import numpy as np
from typing import Tuple
from ..chromatica.gradients.simple_angular_radial import SimpleAngularRadialGradient
from ..chromatica.gradients.full_parametrical_angular_radial import FullParametricalAngularRadialGradient
from ..chromatica.format_type import FormatType

def normalize_angle(angle: float) -> float:
    """Normalize angle to [0, 360) range."""
    return angle % 360.0

def get_simple_gradient_masks(width: int, height: int, radius: float, 
                             deg_start: float, deg_end: float,
                             radius_start: float, radius_end: float,
                             center: Tuple[int, int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the masks used in SimpleAngularRadialGradient for debugging.
    Returns: angular_mask, radial_mask, combined_mask
    """
    if center is None:
        center = (width // 2, height // 2)
    
    # Generate indices np matrix
    indices_matrix = np.indices((height, width), dtype=np.float32)
    
    # Calculate radius and degrees matrices
    y_indices = indices_matrix[0] - center[1]
    x_indices = indices_matrix[1] - center[0]
    
    # Calculate distance from center
    distances = np.sqrt(x_indices**2 + y_indices**2)
    
    # Calculate angles in degrees
    theta = (np.degrees(np.arctan2(y_indices, x_indices)) + 360.0) % 360.0
    
    # Create angular mask using the same logic as the gradient
    angular_mask = np.ones_like(theta, dtype=bool)
    if deg_end >= deg_start:
        angular_range = deg_end - deg_start
        if angular_range >= 360.0:
            angular_mask = np.ones_like(theta, dtype=bool)
        else:
            theta_start = deg_start % 360.0
            theta_end = deg_end % 360.0
            if theta_end >= theta_start:
                angular_mask = (theta >= theta_start) & (theta <= theta_end)
            else:
                angular_mask = (theta >= theta_start) | (theta <= theta_end)
    else:
        theta_start = deg_start % 360.0
        theta_end = deg_end % 360.0
        angular_mask = (theta >= theta_start) | (theta <= theta_end)
    
    # Calculate r_min and r_max based on radius_start and radius_end
    r_min = radius * radius_start
    r_max = radius * radius_end
    
    # Create radial mask
    radial_mask = (distances >= r_min) & (distances <= r_max)
    
    # Combined mask
    combined_mask = angular_mask & radial_mask
    
    return angular_mask, radial_mask, combined_mask

def expanding_ring_masks(n: int = 1, frames: int = 24) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """
    Return masks for debugging: angular mask, radial mask, combined mask
    """
    initial_deg_start = -12
    initial_deg_finish = 12
    width, height = 512, 512
    total_rotation = 180
    this_rotation = (total_rotation / frames) * n
    this_deg_start = max(initial_deg_start - 14 * n // 2, -180) + this_rotation
    this_deg_end = min(initial_deg_finish + 14 * n // 2, 180) + this_rotation
    radius_start = min(0.5 + n / 24 * 0.24, 0.74)
    radius_end = 1.0
    
    center = (width // 2, height // 2)
    radius = 200
    
    # Get masks
    angular_mask, radial_mask, combined_mask = get_simple_gradient_masks(
        width, height, radius, this_deg_start, this_deg_end,
        radius_start, radius_end, center
    )
    
    # Convert masks to images
    angular_img = Image.fromarray((angular_mask * 255).astype(np.uint8), mode='L')
    radial_img = Image.fromarray((radial_mask * 255).astype(np.uint8), mode='L')
    combined_img = Image.fromarray((combined_mask * 255).astype(np.uint8), mode='L')
    
    return angular_img, radial_img, combined_img

def expanding_ring_debug(n: int = 1, frames: int = 24) -> Tuple[Image.Image, dict]:
    """
    Return the gradient and debug information
    """
    initial_deg_start = -12
    initial_deg_finish = 12
    width, height = 512, 512
    total_rotation = 180
    this_rotation = (total_rotation / frames) * n
    this_deg_start = max(initial_deg_start - 14 * n // 2, -180) + this_rotation
    this_deg_end = min(initial_deg_finish + 14 * n // 2, 180) + this_rotation
    radius_start = min(0.5 + n / 24 * 0.24, 0.74)
    radius_end = 1.0
    hue_speed = 6.5
    hue_increase = int(n * hue_speed)
    hue_0 = 176 + hue_increase
    
    # Get masks first
    center = (width // 2, height // 2)
    angular_mask, radial_mask, combined_mask = get_simple_gradient_masks(
        width, height, 200, this_deg_start, this_deg_end,
        radius_start, radius_end, center
    )
    
    # Get the actual gradient
    gradient = SimpleAngularRadialGradient.generate(
        width, height,
        radius=200,
        inner_ring_colors=[(hue_0, 255, 255), (hue_0+20, 255, 255)],
        outer_ring_colors=[(hue_0+20, 255, 255), (hue_0+40, 255, 255)],
        color_space='hsv',
        format_type=FormatType.INT,
        deg_start=this_deg_start,
        deg_end=this_deg_end,
        outside_fill=(128, 0, 255),
        radius_start=radius_start,
        radius_end=radius_end,
    )
    
    gradient_img = gradient.convert('rgb', to_format=FormatType.INT)
    result_img = Image.fromarray(gradient_img.value.astype(np.uint8), mode="RGB")
    
    # Debug info
    debug_info = {
        'n': n,
        'deg_start': this_deg_start,
        'deg_end': this_deg_end,
        'radius_start': radius_start,
        'radius_end': radius_end,
        'r_min': 200 * radius_start,
        'r_max': 200,
        'angular_mask_pct': np.mean(angular_mask) * 100,
        'radial_mask_pct': np.mean(radial_mask) * 100,
        'combined_mask_pct': np.mean(combined_mask) * 100,
        'mask_areas': {
            'angular_white_pixels': np.sum(angular_mask),
            'radial_white_pixels': np.sum(radial_mask),
            'combined_white_pixels': np.sum(combined_mask)
        }
    }
    
    return result_img, debug_info

def visualize_angle_progression():
    """Create a visualization showing how the angle mask changes with n"""
    width, height = 512, 512
    center = (256, 256)
    n_values = [0, 6, 12, 18, 23, 24]
    
    # Create a large canvas
    canvas_width = width * len(n_values)
    canvas_height = height * 2  # Top: masks, Bottom: gradients
    canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
    
    for i, n in enumerate(n_values):
        # Get masks
        angular_img, radial_img, combined_img = expanding_ring_masks(n)
        
        # Get gradient
        gradient_img, debug_info = expanding_ring_debug(n)

        # Convert masks to arrays for drawing
        angular_arr = np.array(angular_img)
        radial_arr = np.array(radial_img)
        
        # Create colored visualization
        mask_visual = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Blue for angular only
        mask_visual[(angular_arr > 0) & (radial_arr == 0)] = [0, 0, 255]
        # Green for radial only  
        mask_visual[(radial_arr > 0) & (angular_arr == 0)] = [0, 255, 0]
        # Cyan for both (intersection)
        mask_visual[(angular_arr > 0) & (radial_arr > 0)] = [0, 255, 255]
        
        mask_img = Image.fromarray(mask_visual)
        
        # Add to canvas
        canvas.paste(mask_img, (i * width, 0))
        canvas.paste(gradient_img, (i * width, height))
        
        # Add labels
        draw = ImageDraw.Draw(canvas)
        draw.text((i * width + 10, 10), f"n={n}", fill=(255, 255, 255))
        draw.text((i * width + 10, height + 10), f"n={n}", fill=(255, 255, 255))
        
        print(f"n={n}: deg_start={debug_info['deg_start']:.1f}, deg_end={debug_info['deg_end']:.1f}, "
              f"mask_area={debug_info['combined_mask_pct']:.1f}%")
    
    canvas.show()
    return canvas

def check_specific_angles():
    """Check what happens at specific angles as n increases"""
    n_values = [0, 6, 12, 18, 23, 24]
    angles_to_check = [0, 90, 180, 270, 355, 359]  # Angles to check
    
    for n in n_values:
        print(f"\n=== n={n} ===")
        
        # Get masks
        angular_img, radial_img, combined_img = expanding_ring_masks(n)
        angular_arr = np.array(angular_img) > 0
        radial_arr = np.array(radial_img) > 0
        combined_arr = np.array(combined_img) > 0
        
        center = (256, 256)
        radius = 200
        
        for angle in angles_to_check:
            # Calculate pixel coordinates
            x = int(center[0] + radius * np.cos(np.radians(angle)))
            y = int(center[1] + radius * np.sin(np.radians(angle)))
            
            # Check if pixel is in each mask
            in_angular = angular_arr[y, x] if 0 <= y < 512 and 0 <= x < 512 else False
            in_radial = radial_arr[y, x] if 0 <= y < 512 and 0 <= x < 512 else False
            in_combined = combined_arr[y, x] if 0 <= y < 512 and 0 <= x < 512 else False
            
            print(f"  Angle {angle:3}°: Angular={in_angular}, Radial={in_radial}, Combined={in_combined}")

def debug_hue_values(n: int = 23):
    """Debug the hue values being generated."""
    # Get the same parameters as expanding_ring
    initial_deg_start = -12
    initial_deg_finish = 12
    width, height = 512, 512
    total_rotation = 180
    frames = 24
    this_rotation = (total_rotation / frames) * n
    hue_speed = 6.5
    hue_increase = int(n * hue_speed)
    hue_0 = 176 + hue_increase
    
    print(f"n={n}: hue_0={hue_0}, hue_0+20={hue_0+20}, hue_0+40={hue_0+40}")
    
    # Create a simple gradient at a specific angle to debug
    center = (256, 256)
    test_angle = 270  # Where we see black
    
    # Calculate pixel coordinates at radius 200
    radius = 200
    x = int(center[0] + radius * np.cos(np.radians(test_angle)))
    y = int(center[1] + radius * np.sin(np.radians(test_angle)))
    
    print(f"Test pixel at angle {test_angle}°, radius {radius}: ({x}, {y})")
    
    # Generate the gradient
    gradient = SimpleAngularRadialGradient.generate(
        width, height,
        radius=200,
        inner_ring_colors=[(hue_0, 255, 255), (hue_0+20, 255, 255)],
        outer_ring_colors=[(hue_0+20, 255, 255), (hue_0+40, 255, 255)],
        color_space='hsv',
        format_type=FormatType.FLOAT,  # Use float to see raw values
        deg_start=initial_deg_start - 14 * n // 2 + this_rotation,
        deg_end=initial_deg_finish + 14 * n // 2 + this_rotation,
        outside_fill=(128, 0, 255),
        radius_start=min(0.5 + n/24*0.24, 0.74),
        radius_end=1.0,
    )
    
    # Get the HSV value at the test pixel
    hsv_value = gradient.value[y, x]
    print(f"Raw HSV value at test pixel: {hsv_value}")
    
    # Check if it's valid HSV
    # Hue should be in [0, 360], Saturation and Value in [0, 255]
    if hsv_value[0] < 0 or hsv_value[0] > 360:
        print(f"⚠️ Invalid hue: {hsv_value[0]}")
    if hsv_value[1] < 0 or hsv_value[1] > 255:
        print(f"⚠️ Invalid saturation: {hsv_value[1]}")
    if hsv_value[2] < 0 or hsv_value[2] > 255:
        print(f"⚠️ Invalid value: {hsv_value[2]}")
    
    # Convert to RGB to see what color it produces
    from colorsys import hsv_to_rgb
    
    # Normalize HSV for colorsys
    h = hsv_value[0] / 360.0
    s = hsv_value[1] / 255.0
    v = hsv_value[2] / 255.0
    
    r, g, b = hsv_to_rgb(h, s, v)
    rgb = (int(r * 255), int(g * 255), int(b * 255))
    
    print(f"Converted RGB: {rgb}")
    print(f"Is black? {rgb == (0, 0, 0)}")
    
    return gradient

def debug_black_spiral(n: int = 23):
    """Examine pixels along the black spiral line."""
    # Generate the gradient
    img, debug_info = expanding_ring_debug(n)
    img_array = np.array(img)
    
    # Find black pixels (RGB = [0, 0, 0])
    black_mask = np.all(img_array == [0, 0, 0], axis=-1)
    black_indices = np.where(black_mask)
    
    if len(black_indices[0]) == 0:
        print(f"No black pixels found for n={n}")
        return None
    
    print(f"Found {len(black_indices[0])} black pixels for n={n}")
    
    # Take a sample of black pixels
    center = (256, 256)
    sample_size = min(20, len(black_indices[0]))
    samples = []
    
    for i in range(0, len(black_indices[0]), len(black_indices[0]) // sample_size):
        if len(samples) >= sample_size:
            break
        y, x = black_indices[0][i], black_indices[1][i]
        
        # Calculate polar coordinates
        dy = y - center[1]
        dx = x - center[0]
        theta = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
        radius = np.sqrt(dx*dx + dy*dy)
        
        # Get masks
        angular_mask, radial_mask, combined_mask = get_simple_gradient_masks(
            512, 512, 200, 
            debug_info['deg_start'], debug_info['deg_end'],
            debug_info['radius_start'], debug_info['radius_end']
        )
        
        samples.append({
            'x': x, 'y': y,
            'theta': theta,
            'radius': radius,
            'in_mask': combined_mask[y, x],
            'pixel_value': img_array[y, x].tolist()
        })
    
    # Analyze the samples
    print("\nSample of black pixels:")
    for i, sample in enumerate(samples):
        print(f"  Pixel {i}: ({sample['x']}, {sample['y']}) - "
              f"θ={sample['theta']:.1f}°, r={sample['radius']:.1f}, "
              f"in_mask={sample['in_mask']}, RGB={sample['pixel_value']}")
    
    return black_mask

def visualize_black_pixels(n: int = 23):
    """Create a visualization showing black pixels."""
    img, debug_info = expanding_ring_debug(n)
    img_array = np.array(img)
    
    # Create an overlay showing black pixels in red
    overlay = img_array.copy()
    black_mask = np.all(img_array == [0, 0, 0], axis=-1)
    overlay[black_mask] = [255, 0, 0]  # Red for black pixels
    
    # Create composite
    composite = Image.fromarray(overlay.astype(np.uint8), mode='RGB')
    
    # Draw some debug info
    draw = ImageDraw.Draw(composite)
    draw.text((10, 10), f"n={n}, black pixels: {np.sum(black_mask)}", fill=(255, 255, 255))
    
    composite.show()
    return composite

def expanding_ring_fixed_hue(n: int = 1, frames: int = 24) -> Image.Image:
    """Fixed version that avoids hue wrapping issues."""
    initial_deg_start = -12
    initial_deg_finish = 12
    width, height = 512, 512
    total_rotation = 180
    this_rotation = (total_rotation / frames) * n
    hue_speed = 6.5
    hue_increase = int(n * hue_speed)
    hue_0 = 176 + hue_increase
    
    # Ensure hue values stay in a safe range (avoid 0°/360° boundary)
    # Keep hue strictly in [0, 360)
    def safe_hue(h):
        return h % 360.0
    
    radius_start = min(0.5 + n / 24 * 0.24, 0.74)
    radius_end = 1.0
    this_deg_start = max(initial_deg_start - 14 * n // 2, -180) + this_rotation
    this_deg_end = min(initial_deg_finish + 14 * n // 2, 180) + this_rotation
    
    gradient = SimpleAngularRadialGradient.generate(
        width, height,
        radius=200,
        inner_ring_colors=[(safe_hue(hue_0), 255, 255), (safe_hue(hue_0 + 20), 255, 255)],
        outer_ring_colors=[(safe_hue(hue_0 + 20), 255, 255), (safe_hue(hue_0 + 40), 255, 255)],
        color_space='hsv',
        format_type=FormatType.INT,
        deg_start=this_deg_start,
        deg_end=this_deg_end,
        outside_fill=(128, 0, 128),
        radius_start=radius_start,
        radius_end=radius_end,
        hue_direction_theta='cw',
        hue_direction_r='cw',
    )
    
    gradient = gradient.convert('rgb', to_format=FormatType.INT)
    return Image.fromarray(gradient.value.astype(np.uint8), mode="RGB")

def expanding_ring(n = 1, frames = 24) -> Image.Image:
    initial_deg_start = -12
    initial_deg_finish = 12
    width, height = 512, 512
    total_rotation = 180
    this_rotation = (total_rotation / frames) * n
    hue_speed = 6.5
    hue_increase = int(n * hue_speed)
    hue_0 = 176 + hue_increase
    radius_start = min(0.5 + n/24*0.24, 0.74)
    radius_end = 1.0
    this_deg_start = max(initial_deg_start-14*n//2, -180) + this_rotation
    this_deg_end = min(initial_deg_finish+ 14*n//2, 180) + this_rotation
    gradient = SimpleAngularRadialGradient.generate(
        width, height,
        radius = 200,
        inner_ring_colors=[(hue_0, 255, 255), (hue_0+20, 255, 255)],
        outer_ring_colors=[(hue_0+20, 255, 255), (hue_0+40, 255, 255)],
        color_space='hsv',
        format_type=FormatType.INT,
        deg_start=this_deg_start,
        deg_end=this_deg_end,
        outside_fill=(128, 0, 255),
        radius_start=radius_start,
        radius_end=radius_end,
        #hue_direction_r="cw",
        #hue_direction_theta="ccw"
        #normalize_radius=False
    )
    #print(gradient.value.dtype, gradient.value)
    gradient = gradient.convert('rgb', to_format=FormatType.INT)
    return Image.fromarray(gradient.value.astype('uint8'), mode="RGB")

def expanding_ring_full_parametrical(n = 1, frames = 24) -> Image.Image:
    initial_deg_start = -12
    initial_deg_finish = 12
    width, height = 512, 512
    total_rotation = 180
    this_rotation = (total_rotation / frames) * n
    hue_speed = 6.5
    hue_increase = int(n * hue_speed)
    hue_0 = 176 + hue_increase
    radius_start = 0.5 + n/24*0.24
    radius_end = 1.0
    this_deg_start = initial_deg_start-14*n//2 + this_rotation
    this_deg_end = initial_deg_finish+ 14*n//2 + this_rotation
    radius = 200
    gradient = FullParametricalAngularRadialGradient.generate(
        width=width,
        height=height,
        inner_r_theta=lambda theta: radius_start*radius,
        outer_r_theta=lambda theta: radius_end*radius,
        color_rings=[
            [(hue_0, 255, 255), (hue_0+20, 255, 255)],
            [(hue_0+20, 255, 255), (hue_0+40, 255, 255)]
        ],
        deg_start=this_deg_start,
        deg_end=this_deg_end,
        color_space='hsv',
        format_type=FormatType.INT,
        outside_fill=(128, 0, 255),
    )
    #print(gradient.value.dtype, gradient.value)
    gradient = gradient.convert('rgb', to_format=FormatType.INT)
    return Image.fromarray(gradient.value.astype('uint8'), mode="RGB")