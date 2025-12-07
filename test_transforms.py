from .chromatica.gradients.full_parametrical_angular_radial import FullParametricalAngularRadialGradient
from .chromatica.format_type import FormatType
import numpy as np
from PIL import Image

def zero_both(r, theta):
    return np.zeros_like(r), np.zeros_like(theta)

def zero_radius(r, theta):
    return np.zeros_like(r), theta

# Test with zero_both transform on alpha channel
print("Testing bivariable_space_transforms with zero_both on alpha channel...")
grad = FullParametricalAngularRadialGradient.generate(
    512, 512,
    inner_r_theta=lambda t: 16*(5+np.sin(8*np.radians(t))),
    outer_r_theta=lambda t: 16*(10+(np.radians(t)+np.sin(8*np.radians(t)))),
    color_rings=[
        ((0,255,0,0),(0,255,255,255)),
        ((255,0,255,255),(255,0,255,0))
    ],
    color_space='rgba',
    format_type=FormatType.INT,
    bivariable_space_transforms={3: zero_both},
    outside_fill=(0,0,0,0)
)

print(f'Alpha channel - min: {np.min(grad.value[:,:,3])}, max: {np.max(grad.value[:,:,3])}')
print(f'Green channel - min: {np.min(grad.value[:,:,1])}, max: {np.max(grad.value[:,:,1])}')

# Test with zero_radius transform on alpha channel
print("\nTesting bivariable_space_transforms with zero_radius on alpha channel...")
grad2 = FullParametricalAngularRadialGradient.generate(
    512, 512,
    inner_r_theta=lambda t: 16*(5+np.sin(8*np.radians(t))),
    outer_r_theta=lambda t: 16*(10+(np.radians(t)+np.sin(8*np.radians(t)))),
    color_rings=[
        ((0,255,0,0),(0,255,255,255)),
        ((255,0,255,255),(255,0,255,0))
    ],
    color_space='rgba',
    format_type=FormatType.INT,
    bivariable_space_transforms={3: zero_radius},
    outside_fill=(0,0,0,0)
)

print(f'Alpha channel - min: {np.min(grad2.value[:,:,3])}, max: {np.max(grad2.value[:,:,3])}')
print(f'Green channel - min: {np.min(grad2.value[:,:,1])}, max: {np.max(grad2.value[:,:,1])}')

# Compare
print(f"\nAre they equal? {np.allclose(grad.value, grad2.value)}")
print("They should be DIFFERENT now that the fix is applied!")

img = Image.fromarray(grad.value.astype(np.uint8), 'RGBA')
#img.show()
img2 = Image.fromarray(grad2.value.astype(np.uint8), 'RGBA')
#img2.show()
