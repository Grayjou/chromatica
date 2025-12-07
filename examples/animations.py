from PIL import Image, ImageDraw
import numpy as np
from typing import Tuple
from ..chromatica.gradients.simple_angular_radial import SimpleAngularRadialGradient
from ..chromatica.gradients.full_parametrical_angular_radial import FullParametricalAngularRadialGradient
from ..chromatica.format_type import FormatType
from PIL import Image

def filling_ring(output_path: str | None = None):
    frames = 24
    n_horizontal = 6
    n_vertical = 4
    def get_row_column(index: int) -> Tuple[int, int]:
        row = index // n_horizontal
        column = index % n_horizontal
        return row, column
    def inner_r_t(progress: float) -> float:
        return 25 + progress * 25
    def outer_r_t(progress: float) -> float:
        return 75 + progress * 24
    cell_width = 200
    cell_height = 200
    h0, s0, v0 = 180, 255, 255  # Cyan
    h1, s1, v1 = 240, 255, 255  # Blue
    total_hue_increase = 120  # degrees
    span_degrees_0 = 60
    span_degrees_1 = 360
    general_offset = -30
    total_offset = 180
    #init_canvas
    canvas = Image.new('RGBA', (cell_width * n_horizontal, cell_height * n_vertical))
    for frame in range(frames):

        progress = frame / (frames - 1)
        this_h0, this_s0, this_v0 = int(h0 + progress * 2 * total_hue_increase), s0, v0
        this_h1, this_s1, this_v1 = int(h1 + progress * total_hue_increase), s1, v1
        this_span = span_degrees_0 + progress * (span_degrees_1 - span_degrees_0)
        this_start_deg = general_offset + progress * total_offset - this_span / 2
        this_end_deg = this_start_deg + this_span
        this_inner_ring = this_outer_ring = (
            (this_h0, this_s0, this_v0),
            (this_h1, this_s1, this_v1)
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
    if output_path:
        canvas.save(output_path)
    else:
        canvas.show()

def filling_ring_problematic(output_path: str | None = None, show: bool = False, total_hue_increase: int = 120):
    frames_to_use = range(18, 24)
    total_frames = 24
    n_horizontal = 6
    n_vertical = 1
    def get_row_column(index: int) -> Tuple[int, int]:
        row = index // n_horizontal
        column = index % n_horizontal
        return row, column
    def inner_r_t(progress: float) -> float:
        return 25 + progress * 25
    def outer_r_t(progress: float) -> float:
        return 75 + progress * 24
    cell_width = 200
    cell_height = 200
    h0, s0, v0 = 180, 255, 255  # Cyan
    h1, s1, v1 = 240, 255, 255  # Blue
    span_degrees_0 = 60
    span_degrees_1 = 360
    general_offset = -30
    total_offset = 180
    #init_canvas
    canvas = Image.new('RGBA', (cell_width * len(frames_to_use), cell_height * n_vertical))
    frames_with_issues = []
    for idx, frame in enumerate(frames_to_use):
        progress = frame / (total_frames - 1)
        this_h0, this_s0, this_v0 = int(h0 + progress * 2 * total_hue_increase), s0, v0
        this_h1, this_s1, this_v1 = int(h1 + progress * total_hue_increase), s1, v1
        this_span = span_degrees_0 + progress * (span_degrees_1 - span_degrees_0)
        this_start_deg = general_offset + progress * total_offset - this_span / 2
        this_end_deg = this_start_deg + this_span
        this_inner_ring = this_outer_ring = (
            (this_h0, this_s0, this_v0),
            (this_h1, this_s1, this_v1)
        )
        #if this_h0 > this_h1:
            #print("Hue wrap-around occurred.", end= " ")
            #print("diff" , this_h0 - this_h1)
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
        # this frame check if black pixels exist (indicating a problem)
        if np.any(np.all(gradient.value == [0, 0, 0], axis=-1)):
            frames_with_issues.append(frame)
            #print(f"Frame {frame} has black pixels indicating an issue.")
        img = Image.fromarray(gradient.value.astype(np.uint8), 'RGB')
        row, column = get_row_column(idx)
        canvas.paste(img, (column * cell_width, row * cell_height))
    print("Frames with issues (black pixels):", frames_with_issues)
    if output_path:
        canvas.save(output_path)
    else:
        canvas.show() if show else None
    return frames_with_issues