from PIL import Image, ImageDraw
import numpy as np
from typing import Tuple
from ..chromatica.gradients.simple_angular_radial import SimpleAngularRadialGradient
from ..chromatica.gradients.full_parametrical_angular_radial import FullParametricalAngularRadialGradient
from ..chromatica.format_type import FormatType
from PIL import Image




def filling_ring(output_path: str | None = None, show: bool = False):
    frames = 24
    n_horizontal = 6
    n_vertical = 4
    def get_row_column(index: int) -> Tuple[int, int]:
        row = index // n_horizontal
        column = index % n_horizontal
        return row, column
    def inner_r_t(progress: float) -> float:
        return 25 + progress * 60
    def outer_r_t(progress: float) -> float:
        return 75 + progress * 24
    cell_width = 200
    cell_height = 200
    h0, s0, v0 = 120, 255, 255  # Cyan
    h1, s1, v1 = 180, 255, 255  # Blue
    total_hue_increase_0 = 360 - h0
    total_hue_increase_1 = 360 - h1
    total_value_change = -127
    span_degrees_0 = 0
    span_degrees_1 = 350
    general_offset = -30
    total_offset = 360
    #init_canvas
    easing = lambda t: np.cos(np.pi*((t-1)/2))**2
    canvas = Image.new('RGBA', (cell_width * n_horizontal, cell_height * n_vertical))
    frame_list = [] 
    for frame in range(frames):

        progress = frame / (frames - 1)
        this_h0, this_s0, this_v0 = int(h0 + progress * total_hue_increase_0), s0, v0
        this_h1, this_s1, this_v1 = int(h1 + progress * total_hue_increase_1), s1, int(v1 + progress * total_value_change)
        this_span = span_degrees_0 + progress * (span_degrees_1 - span_degrees_0)
        this_start_deg = general_offset - this_span / 2 + easing(progress)**2 * total_offset 
        this_end_deg = this_start_deg + this_span
        this_inner_ring = this_outer_ring = (
            (this_h0, this_s0, this_v0),
            (this_h1, this_s1, this_v1),
        )
        print(this_inner_ring, end='// ')
        print("frame", frame, "progress", progress,)
        if this_h0 > this_h1:
            print("Hue wrap-around occurred.", end= " ")
            print("diff" , this_h0 - this_h1)
        this_radius = outer_r_t(progress)
        this_inner_radius = inner_r_t(progress)
        radius_end = 1
        this_radius_start = this_inner_radius / this_radius
        gradient = SimpleAngularRadialGradient.generate(
            width=cell_width,
            height=cell_height,
            radius = this_radius,
            inner_ring_colors=this_inner_ring,
            outer_ring_colors=this_outer_ring,
            color_space='hsv',
            format_type=FormatType.INT,
            deg_start=this_start_deg,
            deg_end=this_end_deg,
            radius_start=this_radius_start,
            radius_end=radius_end,
            outside_fill=(0, 0, 255)
        )
        gradient = gradient.convert('rgb', to_format=FormatType.INT)
        img = Image.fromarray(gradient.value.astype(np.uint8), 'RGB')
        row, column = get_row_column(frame)
        canvas.paste(img, (column * cell_width, row * cell_height))
        frame_list.append(img)
    if output_path:
        canvas.save(output_path)
    else:
        canvas.show() if show else None
    return frame_list

def filling_ring_followup(output_path: str | None = None, show: bool = False):
    frames = 24
    n_horizontal = 6
    n_vertical = 4

    def get_row_column(index: int) -> Tuple[int, int]:
        row = index // n_horizontal
        column = index % n_horizontal
        return row, column

    cell_width = 200
    cell_height = 200

    # --- Values taken from last frame of first 24 ---
    # You printed these in the loop already, so plug the last printed values here:
    last_h0 = 180 + (23/23) * (360 - 180)
    last_h1 = 240 + (23/23) * (360 - 240)
    last_v0 = 255
    last_v1 = int(255 + (23/23) * (-127))   # → 128

    last_span = 350                         # your final span from frame 23
    last_inner_radius = 25 + 1 * 60         # → 85
    last_outer_radius = 75 + 1 * 24         # → 99

    # Transition value: we equalize inner + outer toward the SAME target
    target_value = 200   # Or choose any equalizing value you want

    # Rotation: add +360° over these 24 frames
    starting_offset = 175             # last frame offset +360
    rotation_amount = 360                   # another full spin
    frame_list = []
    canvas = Image.new("RGBA", (cell_width * n_horizontal,
                                cell_height * n_vertical))

    for i in range(frames):
        progress = i / (frames - 1)

        # Hue stays frozen
        this_h0 = int(last_h0)
        this_h1 = int(last_h1)

        # Value equalizes
        this_v0 = int(last_v0 + (target_value - last_v0) * progress)
        this_v1 = int(last_v1 + (target_value - last_v1) * progress)

        # Arc span collapses to 0
        this_span = last_span * (1 - progress)

        # Rotation continues smoothly
        this_start_deg = starting_offset + rotation_amount * progress
        this_end_deg = this_start_deg + this_span

        this_inner_ring = (
            (this_h0, 255, this_v0),
            (this_h1, 255, this_v1),
        )
        this_outer_ring = this_inner_ring

        radius = last_outer_radius
        radius_start = last_inner_radius / radius
        radius_end = 1

        gradient = SimpleAngularRadialGradient.generate(
            width=cell_width,
            height=cell_height,
            radius=radius,
            inner_ring_colors=this_inner_ring,
            outer_ring_colors=this_outer_ring,
            color_space='hsv',
            format_type=FormatType.INT,
            deg_start=this_start_deg,
            deg_end=this_end_deg,
            radius_start=radius_start,
            radius_end=radius_end,
            outside_fill=(0, 0, 255),
        )

        img = Image.fromarray(gradient.convert('rgb', to_format=FormatType.INT).value.astype(np.uint8))

        row, col = get_row_column(i)
        canvas.paste(img, (col * cell_width, row * cell_height))
        frame_list.append(img)
    if output_path:
        canvas.save(output_path)
    else:
        canvas.show() if show else None
    return frame_list

def filling_ring_rgba(output_path: str | None = None, show: bool = False):
    frames = 24
    n_horizontal = 6
    n_vertical = 4
    def get_row_column(index: int) -> Tuple[int, int]:
        row = index // n_horizontal
        column = index % n_horizontal
        return row, column
    def inner_r_t(progress: float) -> float:
        return 25 + progress * 60
    def outer_r_t(progress: float) -> float:
        return 70 + progress * 19
    cell_width = 200
    cell_height = 200
    h0, s0, v0, a0 = 120, 255, 255, 0  # Cyan, transparent
    h1, s1, v1, a1 = 180, 255, 255, 255  # Blue, opaque
    total_hue_increase_0 = 360 - h0
    total_hue_increase_1 = 360 - h1
    total_value_change = -127
    total_alpha_change = 255
    span_degrees_0 = 0
    span_degrees_1 = 350
    general_offset = -30
    total_offset = 360
    easing = lambda t: np.cos(np.pi*((t-1)/2))**2
    canvas = Image.new('RGBA', (cell_width * n_horizontal, cell_height * n_vertical), (0, 0, 0, 0))
    frame_list = [] 
    for frame in range(frames):
        progress = frame / (frames - 1)
        this_h0 = int(h0 + progress * total_hue_increase_0)
        this_s0 = s0
        this_v0 = v0
        this_a0 = int(a0 + progress * total_alpha_change)
        this_h1 = int(h1 + progress * total_hue_increase_1)
        this_s1 = s1
        this_v1 = int(v1 + progress * total_value_change)
        this_a1 = a1
        this_span = span_degrees_0 + progress * (span_degrees_1 - span_degrees_0)
        this_start_deg = general_offset - this_span / 2 + easing(progress)**2 * total_offset 
        this_end_deg = this_start_deg + this_span
        this_inner_ring = (
            (this_h0, this_s0, this_v0, this_a0),
            (this_h1, this_s1, this_v1, this_a1),
        )
        this_outer_ring = this_inner_ring
        this_radius = outer_r_t(progress)
        this_inner_radius = inner_r_t(progress)
        radius_end = 1
        this_radius_start = this_inner_radius / this_radius
        gradient = SimpleAngularRadialGradient.generate(
            width=cell_width,
            height=cell_height,
            radius=this_radius,
            inner_ring_colors=this_inner_ring,
            outer_ring_colors=this_outer_ring,
            color_space='hsva',
            format_type=FormatType.INT,
            deg_start=this_start_deg,
            deg_end=this_end_deg,
            radius_start=this_radius_start,
            radius_end=radius_end,
            outside_fill=(0, 0, 0, 0)
        )
        gradient = gradient.convert('rgba', to_format=FormatType.INT)
        img = Image.fromarray(gradient.value.astype(np.uint8), 'RGBA')
        row, column = get_row_column(frame)
        canvas.paste(img, (column * cell_width, row * cell_height), img)
        frame_list.append(img)
    if output_path:
        canvas.save(output_path)
    else:
        canvas.show() if show else None
    return frame_list

def filling_ring_followup_rgba(output_path: str | None = None, show: bool = False):
    frames = 24
    n_horizontal = 6
    n_vertical = 4

    def get_row_column(index: int) -> Tuple[int, int]:
        row = index // n_horizontal
        column = index % n_horizontal
        return row, column

    cell_width = 200
    cell_height = 200

    # Values taken from last frame of filling_ring_rgba
    last_h0 = 180 + (23/23) * (360 - 180)
    last_h1 = 240 + (23/23) * (360 - 240)
    last_v0 = 255
    last_v1 = int(255 + (23/23) * (-127))
    last_a0 = int(0 + (23/23) * 255)  # → 255
    last_a1 = 255

    last_span = 350
    last_inner_radius = 25 + 1 * 60  # → 85
    last_outer_radius = 70 + 1 * 19  # → 89

    # Transition value: equalize inner + outer toward target
    target_value = 200
    target_alpha = 255  # Keep fully opaque

    # Rotation: add +360° over these 24 frames
    starting_offset = 175
    rotation_amount = 360
    
    frame_list = []
    canvas = Image.new("RGBA", (cell_width * n_horizontal,
                                cell_height * n_vertical), (0, 0, 0, 0))

    for i in range(frames):
        progress = i / (frames - 1)

        # Hue stays frozen
        this_h0 = int(last_h0)
        this_h1 = int(last_h1)

        # Value equalizes
        this_v0 = int(last_v0 + (target_value - last_v0) * progress)
        this_v1 = int(last_v1 + (target_value - last_v1) * progress)

        # Alpha equalizes
        this_a0 = int(last_a0 + (target_alpha - last_a0) * progress)
        this_a1 = int(last_a1 + (target_alpha - last_a1) * progress)

        # Arc span collapses to 0
        this_span = last_span * (1 - progress)

        # Rotation continues smoothly
        this_start_deg = starting_offset + rotation_amount * progress
        this_end_deg = this_start_deg + this_span

        this_inner_ring = (
            (this_h0, 255, this_v0, this_a0),
            (this_h1, 255, this_v1, this_a1),
        )
        this_outer_ring = this_inner_ring

        radius = last_outer_radius
        radius_start = last_inner_radius / radius
        radius_end = 1

        gradient = SimpleAngularRadialGradient.generate(
            width=cell_width,
            height=cell_height,
            radius=radius,
            inner_ring_colors=this_inner_ring,
            outer_ring_colors=this_outer_ring,
            color_space='hsva',
            format_type=FormatType.INT,
            deg_start=this_start_deg,
            deg_end=this_end_deg,
            radius_start=radius_start,
            radius_end=radius_end,
            outside_fill=(0, 0, 0, 0),
        )

        img = Image.fromarray(gradient.convert('rgba', to_format=FormatType.INT).value.astype(np.uint8), 'RGBA')

        row, col = get_row_column(i)
        canvas.paste(img, (col * cell_width, row * cell_height), img)
        frame_list.append(img)
    
    if output_path:
        canvas.save(output_path)
    else:
        canvas.show() if show else None
    return frame_list

WHITE_OUTLINE_WIDTH = 4
D_WHITE_OUTLINE_WIDTH = -2
def filling_ring_white(output_path: str | None = None, show: bool = False):
    frames = 24
    n_horizontal = 6
    n_vertical = 4
    def get_row_column(index: int) -> Tuple[int, int]:
        row = index // n_horizontal
        column = index % n_horizontal
        return row, column
    def inner_r_t(progress: float) -> float:
        return 25 + progress * 60 - WHITE_OUTLINE_WIDTH
    def outer_r_t(progress: float) -> float:
        return 70 + progress * 19 + WHITE_OUTLINE_WIDTH
    cell_width = 200
    cell_height = 200
    h0, s0, v0, a0 = 0, 0, 255, 0  # White, transparent
    h1, s1, v1, a1 = 0, 0, 255, 255  # White, opaque
    total_alpha_change = 255
    span_degrees_0 = 0
    span_degrees_1 = 350
    general_offset = -30
    total_offset = 360
    easing = lambda t: np.cos(np.pi*((t-1)/2))**2
    canvas = Image.new('RGBA', (cell_width * n_horizontal, cell_height * n_vertical), (0, 0, 0, 0))
    frame_list = [] 
    for frame in range(frames):
        progress = frame / (frames - 1)
        this_h0 = h0
        this_s0 = s0
        this_v0 = v0
        this_a0 = int(a0 + progress * total_alpha_change)
        this_h1 = h1
        this_s1 = s1
        this_v1 = v1
        this_a1 = a1
        this_span = span_degrees_0 + progress * (span_degrees_1 - span_degrees_0)
        this_start_deg = general_offset - this_span / 2 + easing(progress)**2 * total_offset 
        this_end_deg = this_start_deg + this_span
        this_inner_ring = (
            (this_h0, this_s0, this_v0, this_a0),
            (this_h1, this_s1, this_v1, this_a1),
        )
        this_outer_ring = this_inner_ring
        this_radius = outer_r_t(progress) + D_WHITE_OUTLINE_WIDTH*progress
        this_inner_radius = inner_r_t(progress) - D_WHITE_OUTLINE_WIDTH*progress
        radius_end = 1
        this_radius_start = this_inner_radius / this_radius
        gradient = SimpleAngularRadialGradient.generate(
            width=cell_width,
            height=cell_height,
            radius=this_radius,
            inner_ring_colors=this_inner_ring,
            outer_ring_colors=this_outer_ring,
            color_space='hsva',
            format_type=FormatType.INT,
            deg_start=this_start_deg,
            deg_end=this_end_deg,
            radius_start=this_radius_start,
            radius_end=radius_end,
            outside_fill=(0, 0, 0, 0)
        )
        gradient = gradient.convert('rgba', to_format=FormatType.INT)
        img = Image.fromarray(gradient.value.astype(np.uint8), 'RGBA')
        row, column = get_row_column(frame)
        canvas.paste(img, (column * cell_width, row * cell_height), img)
        frame_list.append(img)
    if output_path:
        canvas.save(output_path)
    else:
        canvas.show() if show else None
    return frame_list

def filling_ring_followup_white(output_path: str | None = None, show: bool = False):
    frames = 24
    n_horizontal = 6
    n_vertical = 4

    def get_row_column(index: int) -> Tuple[int, int]:
        row = index // n_horizontal
        column = index % n_horizontal
        return row, column

    cell_width = 200
    cell_height = 200

    # Values taken from last frame of filling_ring_white
    last_h0 = 0
    last_h1 = 0
    last_v0 = 255
    last_v1 = 255
    last_a0 = 255
    last_a1 = 255

    last_span = 350
    last_inner_radius = 25 + 1 * 60 - WHITE_OUTLINE_WIDTH - D_WHITE_OUTLINE_WIDTH # → 80
    last_outer_radius = 70 + 1 * 19 + WHITE_OUTLINE_WIDTH + D_WHITE_OUTLINE_WIDTH # → 99

    target_value = 255  # Keep white
    target_alpha = 255

    starting_offset = 175
    rotation_amount = 360
    
    frame_list = []
    canvas = Image.new("RGBA", (cell_width * n_horizontal,
                                cell_height * n_vertical), (0, 0, 0, 0))

    for i in range(frames):
        progress = i / (frames - 1)

        this_h0 = last_h0
        this_h1 = last_h1
        this_v0 = last_v0
        this_v1 = last_v1
        this_a0 = last_a0
        this_a1 = last_a1

        # Arc span collapses to 0
        this_span = last_span * (1 - progress)

        # Rotation continues smoothly
        this_start_deg = starting_offset + rotation_amount * progress
        this_end_deg = this_start_deg + this_span

        this_inner_ring = (
            (this_h0, 0, this_v0, this_a0),
            (this_h1, 0, this_v1, this_a1),
        )
        this_outer_ring = this_inner_ring

        radius = last_outer_radius
        radius_start = last_inner_radius / radius
        radius_end = 1

        gradient = SimpleAngularRadialGradient.generate(
            width=cell_width,
            height=cell_height,
            radius=radius,
            inner_ring_colors=this_inner_ring,
            outer_ring_colors=this_outer_ring,
            color_space='hsva',
            format_type=FormatType.INT,
            deg_start=this_start_deg,
            deg_end=this_end_deg,
            radius_start=radius_start,
            radius_end=radius_end,
            outside_fill=(0, 0, 0, 0),
        )

        img = Image.fromarray(gradient.convert('rgba', to_format=FormatType.INT).value.astype(np.uint8), 'RGBA')

        row, col = get_row_column(i)
        canvas.paste(img, (col * cell_width, row * cell_height), img)
        frame_list.append(img)
    
    if output_path:
        canvas.save(output_path)
    else:
        canvas.show() if show else None
    return frame_list

def fillin_ring_rgba_white_outline(output_path: str | None = None):
    white_startup_frames = filling_ring_white()
    rgba_frames = filling_ring_rgba()
    white_followup_frames = filling_ring_followup_white()
    rgba_followup_frames = filling_ring_followup_rgba()
    combineds = []
    for white_img, rgba_img in zip(white_startup_frames, rgba_frames):
        combined = Image.alpha_composite(white_img, rgba_img)
        if output_path:
            combined.save(f"{output_path}_startup_{white_startup_frames.index(white_img):02}.png")
        combineds.append(combined)
    for white_img, rgba_img in zip(white_followup_frames, rgba_followup_frames):
        combined = Image.alpha_composite(white_img, rgba_img)
        if output_path:
            combined.save(f"{output_path}_followup_{white_followup_frames.index(white_img):02}.png")
        combineds.append(combined)
    return combineds