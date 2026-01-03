import numpy as np
import pytest
from typing import cast

from ....chromatica.gradients.gradient2dv2.generators.cell_lines import LinesCellFactory
from ....chromatica.gradients.gradient2dv2.cell.lines import LinesCell
from ....chromatica.gradients.gradient2dv2.partitions import PerpendicularPartition, PartitionInterval, IndexRoundingMode
from ....chromatica.types.color_types import ColorMode, HueDirection
from ....chromatica.types.format_type import FormatType
from ....chromatica.conversions import np_convert
from boundednumbers import BoundType
from unitfield import upbm_2d
from ....chromatica.gradients.gradient2dv2.helpers import LineInterpMethods


class TestLinesCellFactory:
    def test_init(self):
        """Test basic initialization and lazy cell creation."""
        top_line = np.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        bottom_line = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0]
        ])
        
        factory = LinesCellFactory(
            width=3,
            height=4,
            top_line=top_line,
            bottom_line=bottom_line,
            color_mode=ColorMode.RGB,
            hue_direction_x=HueDirection.CW,
            hue_direction_y=HueDirection.CW,
            line_method=LineInterpMethods.LINES_CONTINUOUS,
            input_format=FormatType.FLOAT,
            boundtypes=BoundType.CLAMP,
            border_mode=None,
            border_value=None,
        )
        
        assert isinstance(factory, LinesCellFactory)
        assert factory._cell is None  # Cell is initially None
        
        # Get cell (lazy creation)
        cell = factory.get_cell()
        assert isinstance(cell, LinesCell)
        assert factory._cell is cell  # Cell is now set
        
        # Check properties are synced
        assert np.array_equal(factory.top_line, top_line)
        assert np.array_equal(factory.bottom_line, bottom_line)
        assert factory.width == 3
        assert factory.height == 4
        assert factory.color_mode == ColorMode.RGB
        assert factory.hue_direction_x == HueDirection.CW
        assert factory.hue_direction_y == HueDirection.CW
        assert factory.line_method == LineInterpMethods.LINES_CONTINUOUS
    
    def test_property_synchronization(self):
        """Test that property changes sync with the underlying cell."""
        top_line = np.array([[1.0, 0, 0], [0.5, 0, 0], [0.0, 0, 0]])
        bottom_line = np.array([[0, 1.0, 0], [0, 0.5, 0], [0, 0.0, 0]])
        
        factory = LinesCellFactory(
            width=3, height=4,
            top_line=top_line, bottom_line=bottom_line,
            color_mode=ColorMode.RGB,
            input_format=FormatType.FLOAT,
        )
        
        # Create cell first
        cell = factory.get_cell()
        
        # Change properties and verify sync
        new_top_line = np.array([[0.5, 0.5, 0], [0.25, 0.25, 0], [0.0, 0.0, 0]])
        factory.top_line = new_top_line
        assert np.array_equal(cell.top_line, new_top_line)
        
        # Verify cache invalidation
        assert cell._value is None  # Cache should be invalidated
        
        # Test bottom line sync
        new_bottom_line = np.array([[0, 0.5, 0.5], [0, 0.25, 0.25], [0, 0.0, 0.0]])
        factory.bottom_line = new_bottom_line
        assert np.array_equal(cell.bottom_line, new_bottom_line)
        
        # Test other property syncs
        factory.hue_direction_x = HueDirection.CCW
        assert cell.hue_direction_x == HueDirection.CCW
        
        factory.hue_direction_y = HueDirection.CW
        assert cell.hue_direction_y == HueDirection.CW
        
        factory.line_method = LineInterpMethods.LINES_DISCRETE
        assert cell.line_method == LineInterpMethods.LINES_DISCRETE
    
    def test_get_value(self):
        """Test value retrieval with and without cell initialization."""
        top_line = np.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        bottom_line = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0]
        ])
        
        factory = LinesCellFactory(
            width=3, height=4,
            top_line=top_line, bottom_line=bottom_line,
            color_mode=ColorMode.RGB,
            input_format=FormatType.FLOAT,
        )
        
        # Test without initialization
        value = factory.get_value()
        assert value is not None
        assert value.shape == (4, 3, 3)  # height, width, channels
        
        # Test specific values
        # Top row should match top_line
        assert np.allclose(value[0], top_line)
        # Bottom row should match bottom_line
        assert np.allclose(value[-1], bottom_line)
    
    def test_per_channel_coords(self):
        """Test per-channel coordinates handling."""
        top_line = np.array([[1.0, 0, 0], [0.5, 0, 0], [0.0, 0, 0]])
        bottom_line = np.array([[0, 1.0, 0], [0, 0.5, 0], [0, 0.0, 0]])
        
        factory = LinesCellFactory(
            width=3, height=4,
            top_line=top_line, bottom_line=bottom_line,
            color_mode=ColorMode.RGB,
            input_format=FormatType.FLOAT,
        )
        
        # Get default coordinates
        pcc = factory.per_channel_coords

        assert pcc.shape == (4, 3, 2)  # height, width
        
        # Set custom coordinates
        new_pcc = [upbm_2d(width=2, height=2) for _ in range(3)]
        factory.per_channel_coords = new_pcc
        
        # Verify coordinates are set
        cell = factory.get_cell()
        assert cell.per_channel_coords == new_pcc
        
        # Test invalid coordinates type
        with pytest.raises(TypeError):
            factory.per_channel_coords = "invalid"  # type: ignore
    
    def test_per_channel_transforms(self):
        """Test per-channel transforms."""
        top_line = np.array([[1.0, 0, 0], [0.5, 0, 0], [0.0, 0, 0]])
        bottom_line = np.array([[0, 1.0, 0], [0, 0.5, 0], [0, 0.0, 0]])
        
        factory = LinesCellFactory(
            width=3, height=4,
            top_line=top_line, bottom_line=bottom_line,
            color_mode=ColorMode.RGB,
            input_format=FormatType.FLOAT,
        )
        
        # Define transform to reverse coordinates
        def reverse_transform(coords: np.ndarray) -> np.ndarray:
            return 1.0 - coords
        
        # Apply transforms
        factory.per_channel_transforms = {0: reverse_transform, 1: reverse_transform, 2: reverse_transform}
        
        # Get value with transforms
        value = factory.get_value()
        assert value is not None
        
        # Without transforms, value would be different
        factory2 = LinesCellFactory(
            width=3, height=4,
            top_line=top_line, bottom_line=bottom_line,
            color_mode=ColorMode.RGB,
            input_format=FormatType.FLOAT,
        )
        value2 = factory2.get_value()
        
        # Values should be different due to transforms
        assert not np.allclose(value, value2)
    
    def test_dimension_changes(self):
        """Test that width/height changes invalidate the cell."""
        top_line = np.array([[1.0, 0, 0], [0.5, 0, 0]])
        bottom_line = np.array([[0, 1.0, 0], [0, 0.5, 0]])
        
        factory = LinesCellFactory(
            width=2, height=3,
            top_line=top_line, bottom_line=bottom_line,
            color_mode=ColorMode.RGB,
            input_format=FormatType.FLOAT,
        )
        
        # Create cell
        cell = factory.get_cell()
        assert cell is not None
        
        # Change width - should invalidate cell
        factory.width = 4
        assert factory._cell is None
        
        # Change height - should invalidate cell
        factory.height = 5
        assert factory._cell is None
    
    def test_color_mode_conversion(self):
        """Test color space conversion."""
        # Start with RGB colors
        top_line = np.array([[1.0, 0, 0], [0.5, 0, 0], [0.0, 0, 0]])
        bottom_line = np.array([[0, 1.0, 0], [0, 0.5, 0], [0, 0.0, 0]])
        
        factory = LinesCellFactory(
            width=3, height=4,
            top_line=top_line, bottom_line=bottom_line,
            color_mode=ColorMode.RGB,
            input_format=FormatType.FLOAT,
        )
        
        # Convert to HSV
        factory.color_mode = ColorMode.HSV
        
        # Verify conversion happened
        assert factory.color_mode == ColorMode.HSV
        #Gotta set hue_direction_y if color space is hue space
        factory.hue_direction_y = HueDirection.CCW
        # Get value in HSV space
        value = factory.get_value()
        assert value is not None
        assert value.shape == (4, 3, 3)
    
    def test_copy_with(self):
        """Test the copy_with method."""
        top_line = np.array([[1.0, 0, 0], [0.5, 0, 0], [0.0, 0, 0]])
        bottom_line = np.array([[0, 1.0, 0], [0, 0.5, 0], [0, 0.0, 0]])
        
        factory = LinesCellFactory(
            width=3, height=4,
            top_line=top_line, bottom_line=bottom_line,
            color_mode=ColorMode.RGB,
            input_format=FormatType.FLOAT,
            hue_direction_x=HueDirection.CW,
            hue_direction_y=HueDirection.CCW,
            line_method=LineInterpMethods.LINES_CONTINUOUS,
            boundtypes=BoundType.CLAMP,
        )
        
        # Create a copy with modified properties
        copy_factory = factory.copy_with(
            width=5,
            height=6,
            hue_direction_x=HueDirection.CCW,
            line_method=LineInterpMethods.LINES_DISCRETE,
        )
        
        # Verify original unchanged
        assert factory.width == 3
        assert factory.height == 4
        assert factory.hue_direction_x == HueDirection.CW
        assert factory.line_method == LineInterpMethods.LINES_CONTINUOUS
        
        # Verify copy has new values
        assert copy_factory.width == 5
        assert copy_factory.height == 6
        assert copy_factory.hue_direction_x == HueDirection.CCW
        assert copy_factory.line_method == LineInterpMethods.LINES_DISCRETE
        
        # Verify some properties remain the same
        assert copy_factory.color_mode == factory.color_mode
        assert copy_factory.hue_direction_y == factory.hue_direction_y
        assert np.array_equal(copy_factory.top_line, factory.top_line)
        assert np.array_equal(copy_factory.bottom_line, factory.bottom_line)
    
    def test_perpendicular_partition_basic(self):
        """Test basic partition with two intervals."""
        top_line = np.array([
            [1.0, 0.0, 0.0],   # idx 0
            [0.75, 0.0, 0.0],  # idx 1
            [0.5, 0.0, 0.0],   # idx 2 (boundary)
            [0.25, 0.0, 0.0],  # idx 3
            [0.0, 0.0, 0.0]    # idx 4
        ])
        bottom_line = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.75, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, 0.0, 0.0]
        ])

        factory = LinesCellFactory(
            width=5, height=3,
            top_line=top_line, bottom_line=bottom_line,
            color_mode=ColorMode.RGB,
            input_format=FormatType.FLOAT,
        )

        # Create partition: [0, 0.5] = RGB, (0.5, 1] = HSV
        pi1 = PartitionInterval("rgb")
        pi2 = PartitionInterval("hsv", hue_direction_x=HueDirection.CCW, hue_direction_y=HueDirection.CW)
        pp = PerpendicularPartition(breakpoints=[0.5], values=[pi1, pi2])

        # Partition with default padding (1)
        factory1, factory2 = factory.partition_slice(partition=pp)

        # === Dimension checks ===
        # slice_ends = [0, ceil(0.5*4)+1, 5] = [0, 3, 5]
        # factory1: indices [0,1,2] (3 pixels), no padding
        # factory2: indices [3,4] (2 pixels) + 1 left padding = 3 pixels
        assert factory1.width == 3
        assert factory2.width == 3
        assert factory1.height == factory2.height == 3

        # === Color space checks ===
        assert factory1.color_mode == ColorMode.RGB
        assert factory2.color_mode == ColorMode.HSV

        # === Factory1 line checks (RGB, no conversion needed) ===
        # Factory1 gets indices 0, 1, 2
        assert np.array_equal(factory1.top_line, top_line[:3])
        assert np.array_equal(factory1.bottom_line, bottom_line[:3])

        # === Factory2 line checks (HSV, need conversion) ===
        # Factory2 gets: [boundary_idx_2, idx_3, idx_4] converted to HSV
        # Convert back to RGB for comparison
        factory2_top_rgb = np_convert(
            factory2.top_line, 
            from_space=ColorMode.HSV, 
            to_space=ColorMode.RGB,
            input_type='float', 
            output_type='float'
        )
        factory2_bottom_rgb = np_convert(
            factory2.bottom_line,
            from_space=ColorMode.HSV,
            to_space=ColorMode.RGB,
            input_type='float',
            output_type='float'
        )
        
        # Full factory2 line should be [idx2, idx3, idx4]
        assert np.allclose(factory2_top_rgb, top_line[2:])
        assert np.allclose(factory2_bottom_rgb, bottom_line[2:])

        # === Boundary consistency check ===
        # Last column of factory1 should match first column of factory2 (shared boundary)
        # Both should equal top_line[2]
        factory1_boundary = factory1.top_line[-1]  # RGB
        factory2_boundary_rgb = factory2_top_rgb[0]  # Converted to RGB
        
        assert np.allclose(factory1_boundary, top_line[2])
        assert np.allclose(factory2_boundary_rgb, top_line[2])
        assert np.allclose(factory1_boundary, factory2_boundary_rgb)


    def test_perpendicular_partition_multiple_intervals(self):
        """Test partition with three intervals and verify boundary consistency."""
        # Create a simple linear gradient for easy verification
        # Values decrease: 1.0 -> 0.0 across 6 pixels
        top_line = np.array([
            [1.0, 0.0, 0.0],  # idx 0
            [0.8, 0.0, 0.0],  # idx 1
            [0.6, 0.0, 0.0],  # idx 2 (boundary 1)
            [0.4, 0.0, 0.0],  # idx 3
            [0.2, 0.0, 0.0],  # idx 4 (boundary 2)
            [0.0, 0.0, 0.0]   # idx 5
        ])
        bottom_line = top_line.copy()

        factory = LinesCellFactory(
            width=6, height=2,
            top_line=top_line, bottom_line=bottom_line,
            color_mode=ColorMode.RGB,
            input_format=FormatType.FLOAT,
        )

        # Three intervals: RGB -> HSV -> RGB
        pi1 = PartitionInterval("rgb")
        pi2 = PartitionInterval("hsv", hue_direction_y=HueDirection.CCW)
        pi3 = PartitionInterval("rgb")
        pp = PerpendicularPartition(breakpoints=[1/3, 2/3], values=[pi1, pi2, pi3])

        # Partition with CEIL rounding and default padding=1
        # breakpoints at 1/3 and 2/3 of (width-1)=5:
        #   1/3 * 5 = 1.67 -> ceil = 2, +1 = 3 (slice_end for interval 0)
        #   2/3 * 5 = 3.33 -> ceil = 4, +1 = 5 (slice_end for interval 1)
        # slice_ends = [0, 3, 5, 6]
        factory1, factory2, factory3 = factory.partition_slice(
            partition=pp, 
            index_rounding_mode=IndexRoundingMode.CEIL
        )

        # === Color space checks ===
        assert factory1.color_mode == ColorMode.RGB
        assert factory2.color_mode == ColorMode.HSV
        assert factory3.color_mode == ColorMode.RGB

        # === Dimension checks ===
        # Base widths: [3, 2, 1]
        # With padding=1: extra_left=0, extra_right=1
        #   factory1: 3 + 0 = 3
        #   factory2: 2 + 0 + 1 = 3 (gets padding on left)
        #   factory3: 1 + 1 = 2 (gets padding on left)
        assert factory1.width == 3
        assert factory2.width == 3
        assert factory3.width == 2

        # === Helper to convert to RGB for comparison ===
        def to_rgb(line: np.ndarray, from_space: ColorMode) -> np.ndarray:
            if from_space == ColorMode.RGB:
                return line
            return np_convert(
                line, from_space=from_space, to_space=ColorMode.RGB,
                input_type='float', output_type='float'
            )

        # === Boundary consistency checks ===
        # Boundary between factory1 and factory2 should be idx 2 (value [0.6, 0, 0])
        boundary_1_2_expected = top_line[2]
        
        factory1_right_boundary = factory1.top_line[-1]  # RGB
        factory2_left_boundary = to_rgb(factory2.top_line[0], factory2.color_mode)
        
        assert np.allclose(factory1_right_boundary, boundary_1_2_expected), \
            f"Factory1 right boundary mismatch: {factory1_right_boundary} vs {boundary_1_2_expected}"
        assert np.allclose(factory2_left_boundary, boundary_1_2_expected), \
            f"Factory2 left boundary mismatch: {factory2_left_boundary} vs {boundary_1_2_expected}"

        # Boundary between factory2 and factory3 should be idx 4 (value [0.2, 0, 0])
        boundary_2_3_expected = top_line[4]
        
        factory2_right_boundary = to_rgb(factory2.top_line[-1], factory2.color_mode)
        factory3_left_boundary = factory3.top_line[0]  # RGB
        
        assert np.allclose(factory2_right_boundary, boundary_2_3_expected), \
            f"Factory2 right boundary mismatch: {factory2_right_boundary} vs {boundary_2_3_expected}"
        assert np.allclose(factory3_left_boundary, boundary_2_3_expected), \
            f"Factory3 left boundary mismatch: {factory3_left_boundary} vs {boundary_2_3_expected}"

        # === Value boundary consistency ===
        # Rendered values at boundaries should match
        value1 = factory1.get_value()
        value2_rgb = np_convert(
            factory2.get_value(), 
            from_space=factory2.color_mode, 
            to_space=ColorMode.RGB,
            input_type='float', 
            output_type='float'
        )
        value3 = factory3.get_value()

        # Last column of value1 should match first column of value2
        assert np.allclose(value1[:, -1, :], value2_rgb[:, 0, :]), \
            f"Value boundary 1-2 mismatch:\n{value1[:, -1, :]}\nvs\n{value2_rgb[:, 0, :]}"

        # Last column of value2 should match first column of value3
        assert np.allclose(value2_rgb[:, -1, :], value3[:, 0, :]), \
            f"Value boundary 2-3 mismatch:\n{value2_rgb[:, -1, :]}\nvs\n{value3[:, 0, :]}"
    
    def test_perpendicular_partition_padding_variations(self):
        """Test partition with different padding values."""
        top_line = np.array([
            [1.0, 0.0, 0.0],
            [0.75, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        bottom_line = top_line.copy()  # Same for simplicity
        
        factory = LinesCellFactory(
            width=5, height=3,
            top_line=top_line, bottom_line=bottom_line,
            color_mode=ColorMode.RGB,
            input_format=FormatType.FLOAT,
        )
        
        pi1 = PartitionInterval("rgb")
        pi2 = PartitionInterval("rgb")
        pp = PerpendicularPartition(breakpoints=[0.5], values=[pi1, pi2])
        
        # Test padding=0
        factory1, factory2 = factory.partition_slice(partition=pp, padding=0)
        assert factory1.width == 3  # ceil(0.5 * 5)
        assert factory2.width == 2  # floor(0.5 * 5)
        
        # Test padding=2
        factory1, factory2 = factory.partition_slice(partition=pp, padding=2)
        assert factory1.width == 4  # ceil(0.5 * 5) + 1
        assert factory2.width == 3  # floor(0.5 * 5) + 1
        
        # Verify cumulative width
        factory1, factory2 = factory.partition_slice(partition=pp, padding=1)
        base_w1, base_w2 = factory1.width, factory2.width
        
        for i in range(1, 5):
            f1, f2 = factory.partition_slice(partition=pp, padding=2*i+1)
            # Each interval gets i extra columns (distributed between left/right)
            assert f1.width == base_w1 + i
            assert f2.width == base_w2 + i
  
    def test_pure_partition_slice(self):
        """Test partition with pure_partition=True."""
        top_line = np.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        bottom_line = top_line.copy()
        
        factory = LinesCellFactory(
            width=4, height=3,
            top_line=top_line, bottom_line=bottom_line,
            color_mode=ColorMode.RGB,
            input_format=FormatType.FLOAT,
        )
        
        # Get original per-channel coords
        original_pcc = factory.per_channel_coords
        
        pi1 = PartitionInterval("rgb")
        pi2 = PartitionInterval("rgb")
        pp = PerpendicularPartition(breakpoints=[0.5], values=[pi1, pi2])
        
        # Test pure_partition with padding=0
        factory1, factory2 = factory.partition_slice(
            partition=pp,
            pure_partition=True,
            padding=0,
            index_rounding_mode=IndexRoundingMode.FLOOR
        )
        
        # Without padding, per-channel coords shouldn't overlap
        pcc1 = factory1.per_channel_coords
        pcc2 = factory2.per_channel_coords
        
        # They should be slices of original
        if isinstance(pcc1, list):
            # First slice: columns 0-1
            assert pcc1[0].shape[1] == 2  # width of slice
            # Second slice: columns 2-3
            assert pcc2[0].shape[1] == 2  # width of slice
            
            # Verify they are correct slices
            assert np.array_equal(pcc1[0], original_pcc[0][:, :2])
            assert np.array_equal(pcc2[0], original_pcc[0][:, 2:])
        
        # Test with padding=1
        factory1p, factory2p = factory.partition_slice(
            partition=pp,
            pure_partition=True,
            padding=1,
            index_rounding_mode=IndexRoundingMode.FLOOR
        )
        
        # With padding, boundary columns should match
        pcc1p = factory1p.per_channel_coords
        pcc2p = factory2p.per_channel_coords
        
        if isinstance(pcc1p, list):
            # Check that boundary columns match
            assert np.array_equal(pcc1p[0][:, -1:], pcc2p[0][:, :1])
    
    def test_index_rounding_modes(self):
        """Test different index rounding modes."""
        top_line = np.array([
            [1.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        bottom_line = top_line.copy()
        
        factory = LinesCellFactory(
            width=6, height=2,
            top_line=top_line, bottom_line=bottom_line,
            color_mode=ColorMode.RGB,
            input_format=FormatType.FLOAT,
        )
        
        pi1 = PartitionInterval("rgb")
        pi2 = PartitionInterval("rgb")
        pp = PerpendicularPartition(breakpoints=[0.333], values=[pi1, pi2])  # 0.333 * 5 = 1.665
        
        # Test different rounding modes
        for mode in [IndexRoundingMode.FLOOR, IndexRoundingMode.ROUND, IndexRoundingMode.CEIL]:
            factories = factory.partition_slice(
                partition=pp,
                padding=0,
                pure_partition=False,
                index_rounding_mode=mode
            )
            
            # Should always get 2 factories
            assert len(factories) == 2
            
            # Widths depend on rounding
            if mode == IndexRoundingMode.FLOOR:
                # 1.665 -> floor = 1
                assert factories[0].width == 2  # 0-1 inclusive + 1 for boundary
                assert factories[1].width == 4  # 2-5
            elif mode == IndexRoundingMode.ROUND:
                # 1.665 -> round = 2
                assert factories[0].width == 3  # 0-2
                assert factories[1].width == 3  # 3-5
            elif mode == IndexRoundingMode.CEIL:
                # 1.665 -> ceil = 2
                assert factories[0].width == 3  # 0-2
                assert factories[1].width == 3  # 3-5
    
    def test_from_corners(self):
        """Test creating LinesCellFactory from corners."""
        width, height = 5, 4
        
        # Define corners
        top_left = np.array([1.0, 0.0, 0.0])
        top_right = np.array([0.0, 1.0, 0.0])
        bottom_left = np.array([0.0, 0.0, 1.0])
        bottom_right = np.array([1.0, 1.0, 0.0])
        
        # Create from corners
        factory = LinesCellFactory.from_corners(
            width=width,
            height=height,
            top_left=top_left,
            top_right=top_right,
            bottom_left=bottom_left,
            bottom_right=bottom_right,
            color_mode=ColorMode.RGB,
            hue_direction_x=HueDirection.CW,
            hue_direction_y=HueDirection.CW,
            line_method=LineInterpMethods.LINES_CONTINUOUS,
        )
        
        # Verify properties
        assert factory.width == width
        assert factory.height == height
        assert factory.color_mode == ColorMode.RGB
        assert factory.line_method == LineInterpMethods.LINES_CONTINUOUS
        
        # Verify lines are created correctly
        assert factory.top_line.shape == (width, 3)  # width x channels
        assert factory.bottom_line.shape == (width, 3)
        
        # Check line endpoints
        assert np.allclose(factory.top_line[0], top_left)
        assert np.allclose(factory.top_line[-1], top_right)
        assert np.allclose(factory.bottom_line[0], bottom_left)
        assert np.allclose(factory.bottom_line[-1], bottom_right)
        
        # Get value and verify shape
        value = factory.get_value()
        assert value.shape == (height, width, 3)
    
    def test_line_interpolation_methods(self):
        """Test different line interpolation methods."""
        top_line = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        bottom_line = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0]
        ])
        
        for method in [LineInterpMethods.LINES_CONTINUOUS, LineInterpMethods.LINES_DISCRETE]:
            factory = LinesCellFactory(
                width=3, height=4,
                top_line=top_line, bottom_line=bottom_line,
                color_mode=ColorMode.RGB,
                input_format=FormatType.FLOAT,
                line_method=method,
            )
            
            value = factory.get_value()
            assert value is not None
            assert value.shape == (4, 3, 3)
            
            # For discrete method, check that horizontal interpolation is discrete
            if method == LineInterpMethods.LINES_DISCRETE:
                # Top row should match top_line exactly
                assert np.allclose(value[0], top_line)
                # Bottom row should match bottom_line exactly
                assert np.allclose(value[-1], bottom_line)
    
    def test_edge_interpolation(self):
        """Test edge interpolation method."""
        top_line = np.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        bottom_line = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0]
        ])
        
        factory = LinesCellFactory(
            width=3, height=4,
            top_line=top_line, bottom_line=bottom_line,
            color_mode=ColorMode.RGB,
            input_format=FormatType.FLOAT,
            line_method=LineInterpMethods.LINES_CONTINUOUS,
        )
        
        cell = factory.get_cell()
        
        # Test edge interpolation at fraction 0.5
        top_edge = cell.interpolate_edge(0.5, is_top_edge=True)
        bottom_edge = cell.interpolate_edge(0.5, is_top_edge=False)
        
        # At fraction 0.5, should be middle of lines
        assert np.allclose(top_edge, top_line[1])
        assert np.allclose(bottom_edge, bottom_line[1])
        
        # Test edge interpolation at boundaries
        left_edge_top = cell.interpolate_edge(0.0, is_top_edge=True)
        right_edge_top = cell.interpolate_edge(1.0, is_top_edge=True)
        
        assert np.allclose(left_edge_top, top_line[0])
        assert np.allclose(right_edge_top, top_line[-1])
    
    def test_border_modes(self):
        """Test different border modes."""
        top_line = np.array([[1.0, 0, 0], [0.5, 0, 0], [0.0, 0, 0]])
        bottom_line = np.array([[0, 1.0, 0], [0, 0.5, 0], [0, 0.0, 0]])
        
        for border_mode in [None, 0, 1]:  # None, replicate, constant
            factory = LinesCellFactory(
                width=3, height=4,
                top_line=top_line, bottom_line=bottom_line,
                color_mode=ColorMode.RGB,
                input_format=FormatType.FLOAT,
                border_mode=border_mode,
                border_value=0.5 if border_mode == 1 else None,
            )
            
            value = factory.get_value()
            assert value is not None
            
            # Basic shape check
            assert value.shape == (4, 3, 3)
    
    def test_boundtypes(self):
        """Test different boundtypes."""
        top_line = np.array([[1.0, 0, 0], [0.5, 0, 0], [0.0, 0, 0]])
        bottom_line = np.array([[0, 1.0, 0], [0, 0.5, 0], [0, 0.0, 0]])
        
        for boundtype in [BoundType.CLAMP, BoundType.CYCLIC, BoundType.BOUNCE]:
            factory = LinesCellFactory(
                width=3, height=4,
                top_line=top_line, bottom_line=bottom_line,
                color_mode=ColorMode.RGB,
                input_format=FormatType.FLOAT,
                boundtypes=boundtype,
            )
            
            value = factory.get_value()
            assert value is not None
            assert value.shape == (4, 3, 3)
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test mismatched line lengths
        top_line = np.array([[1.0, 0, 0], [0.5, 0, 0]])  # length 2
        bottom_line = np.array([[0, 1.0, 0], [0, 0.5, 0], [0, 0.0, 0]])  # length 3
        
        with pytest.raises(ValueError):
            LinesCellFactory(
                width=3, # LinesCell doesn't care if width matches line length, but it cares if they match each other
                height=4,
                top_line=top_line,
                bottom_line=bottom_line,
                color_mode=ColorMode.RGB,
                input_format=FormatType.FLOAT,
            )
        
        # Test invalid color space
        valid_top_line = np.array([[1.0, 0, 0], [0.5, 0, 0], [0.0, 0, 0]])
        valid_bottom_line = np.array([[0, 1.0, 0], [0, 0.5, 0], [0, 0.0, 0]])
        
        with pytest.raises(ValueError):
            LinesCellFactory(
                width=3,
                height=4,
                top_line=valid_top_line,
                bottom_line=valid_bottom_line,
                color_mode="INVALID_color_mode",  # type: ignore
                input_format=FormatType.FLOAT,
            )
    
    def test_repr_and_eq(self):
        """Test string representation and equality."""
        top_line = np.array([[1.0, 0, 0], [0.5, 0, 0]])
        bottom_line = np.array([[0, 1.0, 0], [0, 0.5, 0]])
        
        factory1 = LinesCellFactory(
            width=2, height=3,
            top_line=top_line, bottom_line=bottom_line,
            color_mode=ColorMode.RGB,
            input_format=FormatType.FLOAT,
        )
        
        factory2 = LinesCellFactory(
            width=2, height=3,
            top_line=top_line.copy(), bottom_line=bottom_line.copy(),
            color_mode=ColorMode.RGB,
            input_format=FormatType.FLOAT,
        )
        
        # Test equality
        assert factory1 == factory2
        
        # Test inequality
        factory3 = LinesCellFactory(
            width=3, height=3,  # Different width
            top_line=np.vstack([top_line, [0.0, 0.0, 0.0]]),
            bottom_line=np.vstack([bottom_line, [0.0, 0.0, 0.0]]),
            color_mode=ColorMode.RGB,
            input_format=FormatType.FLOAT,
        )
        
        assert factory1 != factory3
        
        # Test repr
        repr_str = repr(factory1)
        assert "LinesCellFactory" in repr_str
        assert "width=2" in repr_str
        assert "height=3" in repr_str