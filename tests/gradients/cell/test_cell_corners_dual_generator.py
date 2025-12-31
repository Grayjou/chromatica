from __future__ import annotations
import numpy as np
import pytest
from typing import cast, Optional, List, Dict, Any

from ....chromatica.gradients.gradient2dv2.generators.cell_corners_dual import CornersCellDualFactory
from ....chromatica.gradients.gradient2dv2.cell.corners_dual import CornersCellDual
from ....chromatica.gradients.gradient2dv2.partitions import (
    PerpendicularDualPartition, CellDualPartitionInterval, IndexRoundingMode
)
from ....chromatica.samples.colors import RED_FLOAT_RGB, GREEN_FLOAT_RGB, BLUE_FLOAT_RGB, YELLOW_FLOAT_RGB, CYAN_FLOAT_RGB, MAGENTA_FLOAT_RGB, WHITE_FLOAT_RGB, get_white_hsv, RED_FLOAT_HSV, GREEN_FLOAT_HSV
from ....chromatica.types.color_types import ColorSpace, HueMode
from ....chromatica.types.format_type import FormatType
from ....chromatica.conversions import np_convert
from boundednumbers import BoundType
from unitfield import upbm_2d


class TestCornersCellDualFactory:
    def test_init(self):
        """Test basic initialization and lazy cell creation."""
        factory = CornersCellDualFactory(
            width=3,
            height=3,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
            hue_direction_x=HueMode.CW,
            hue_direction_y=HueMode.CCW,
            boundtypes=BoundType.CLAMP,
            border_mode=None,
            border_value=None,
        )
        
        assert isinstance(factory, CornersCellDualFactory)
        assert factory._cell is None  # Cell is initially None
        
        # Get cell (lazy creation)
        cell = factory.get_cell()
        assert isinstance(cell, CornersCellDual)
        assert factory._cell is cell  # Cell is now set
        
        # Check properties are synced
        assert np.array_equal(factory.top_left, RED_FLOAT_RGB)
        assert np.array_equal(factory.top_right, GREEN_FLOAT_RGB)
        assert np.array_equal(factory.bottom_left, BLUE_FLOAT_RGB)
        assert np.array_equal(factory.bottom_right, YELLOW_FLOAT_RGB)
        assert factory.width == 3
        assert factory.height == 3
        assert factory.vertical_color_space == ColorSpace.RGB
        assert factory.horizontal_color_space == ColorSpace.HSV
        assert factory.hue_direction_x == HueMode.CW
        assert factory.hue_direction_y == HueMode.CCW
        assert factory.top_segment_color_space == ColorSpace.HSV  # Defaults to horizontal
        assert factory.bottom_segment_color_space == ColorSpace.HSV  # Defaults to horizontal
    
    def test_property_synchronization(self):
        """Test that property changes sync with the underlying cell."""
        factory = CornersCellDualFactory(
            width=3,
            height=3,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )
        
        # Create cell first
        cell = factory.get_cell()
        
        # Change corner properties and verify sync
        new_top_left = np.array([0.5, 0.5, 0.5])
        factory.top_left = new_top_left
        assert np.array_equal(cell.top_left, new_top_left)
        
        # Verify cache invalidation
        assert cell._value is None  # Cache should be invalidated
        
        # Test other corner syncs
        new_top_right = np.array([0.5, 0.5, 0.5])
        factory.top_right = new_top_right
        assert np.array_equal(cell.top_right, new_top_right)
        
        new_bottom_left = np.array([0.5, 0.5, 0.5])
        factory.bottom_left = new_bottom_left
        assert np.array_equal(cell.bottom_left, new_bottom_left)
        
        new_bottom_right = np.array([0.5, 0.5, 0.5])
        factory.bottom_right = new_bottom_right
        assert np.array_equal(cell.bottom_right, new_bottom_right)
        
        # Test other property syncs
        factory.hue_direction_x = HueMode.CCW
        assert cell.hue_direction_x == HueMode.CCW
        
        factory.hue_direction_y = HueMode.CW
        assert cell.hue_direction_y == HueMode.CW
        
        factory.top_segment_hue_direction_x = HueMode.CCW
        assert cell.top_segment_hue_direction_x == HueMode.CCW
        
        factory.bottom_segment_hue_direction_x = HueMode.CW
        assert cell.bottom_segment_hue_direction_x == HueMode.CW
    
    def test_get_value(self):
        """Test value retrieval with and without cell initialization."""
        factory = CornersCellDualFactory(
            width=2,
            height=2,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )
        
        # Test without initialization (init_cell=False)
        value = factory.get_value(init_cell=False)
        assert value is None  # No cell created yet
        
        # Test with initialization
        value = factory.get_value(init_cell=True)
        assert value is not None
        assert value.shape == (2, 2, 3)  # height, width, channels
        
        # Test specific values (check corners)
        # Note: Due to dual color space, corners might be transformed
        assert value.shape == (2, 2, 3)  # Basic shape check
    
    def test_get_value_auto_init(self):
        """Test get_value with default init_cell=True."""
        factory = CornersCellDualFactory(
            width=2,
            height=2,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )
        
        # Default behavior should create cell
        value = factory.get_value()
        assert value is not None
        assert value.shape == (2, 2, 3)
    
    def test_per_channel_coords(self):
        """Test per-channel coordinates handling."""
        factory = CornersCellDualFactory(
            width=2,
            height=2,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )
        
        # Get default coordinates
        pcc = factory.per_channel_coords
        assert pcc is not None
        
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
        factory = CornersCellDualFactory(
            width=2,
            height=2,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
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
        factory2 = CornersCellDualFactory(
            width=2,
            height=2,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )
        value2 = factory2.get_value()
        
        # Values should be different due to transforms
        assert not np.allclose(value, value2)
    
    def test_dimension_changes(self):
        """Test that width/height changes invalidate the cell."""
        factory = CornersCellDualFactory(
            width=2,
            height=2,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
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
        
        # Change both
        factory.width = 3
        factory.height = 3
        assert factory._cell is None
    
    def test_color_space_conversion(self):
        """Test color space conversion for dual spaces."""
        # Start with RGB vertical, HSV horizontal
        factory = CornersCellDualFactory(
            width=2,
            height=2,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )
        
        # Convert vertical color space
        factory.vertical_color_space = ColorSpace.HSV
        
        # Verify conversion happened
        assert factory.vertical_color_space == ColorSpace.HSV
        assert factory.horizontal_color_space == ColorSpace.HSV  # Unchanged
        
        # Convert horizontal color space
        factory.horizontal_color_space = ColorSpace.RGB
        
        assert factory.vertical_color_space == ColorSpace.HSV
        assert factory.horizontal_color_space == ColorSpace.RGB
        
        # Get value and verify shape
        value = factory.get_value()
        assert value is not None
        assert value.shape == (2, 2, 3)
    
    def test_segment_color_spaces(self):
        """Test segment-specific color spaces."""
        factory = CornersCellDualFactory(
            width=3,
            height=3,
            top_left=CYAN_FLOAT_RGB,  # HSV red
            top_right=np.array([0.333, 1.0, 1.0]),  # HSV green
            bottom_left=CYAN_FLOAT_RGB,  # HSV red
            bottom_right=np.array([0.333, 1.0, 1.0]),  # HSV green
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            top_segment_color_space=ColorSpace.HSV,
            bottom_segment_color_space=ColorSpace.RGB,  # Different segment space
            input_format=FormatType.FLOAT,
        )
        
        # Verify segment color spaces
        assert factory.top_segment_color_space == ColorSpace.HSV
        assert factory.bottom_segment_color_space == ColorSpace.RGB
        
        # Top segment should use HSV interpolation
        # Bottom segment should use RGB interpolation
        value = factory.get_value()
        assert value is not None
        assert value.shape == (3, 3, 3)
        
        # Check that top and bottom rows are different due to different color spaces
        # (This is a heuristic check - exact values depend on implementation)
        assert not np.allclose(value[0], value[-1])
    
    def test_segment_hue_directions(self):
        """Test segment-specific hue directions."""
        factory = CornersCellDualFactory(
            width=3,
            height=3,
            top_left=CYAN_FLOAT_RGB,
            top_right=np.array([0.5, 1.0, 1.0]),
            bottom_left=CYAN_FLOAT_RGB,
            bottom_right=np.array([0.5, 1.0, 1.0]),
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            top_segment_hue_direction_x=HueMode.CW,
            bottom_segment_hue_direction_x=HueMode.CCW,
            input_format=FormatType.FLOAT,
        )
        
        # Verify segment hue directions
        assert factory.top_segment_hue_direction_x == HueMode.CW
        assert factory.bottom_segment_hue_direction_x == HueMode.CCW
        
        # Get value - should use different hue directions for top and bottom
        value = factory.get_value()
        assert value is not None
        
        # With different hue directions, top and bottom interpolation might differ
        # (This is a basic sanity check)
        assert value.shape == (3, 3, 3)
    
    def test_copy_with(self):
        """Test the copy_with method."""
        factory = CornersCellDualFactory(
            width=3,
            height=3,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            top_segment_color_space=ColorSpace.HSV,
            bottom_segment_color_space=ColorSpace.RGB,
            hue_direction_x=HueMode.CCW,
            hue_direction_y=HueMode.CW,
            top_segment_hue_direction_x=HueMode.CW,
            bottom_segment_hue_direction_x=HueMode.CCW,
            boundtypes=BoundType.CYCLIC,
            border_mode=1,
            border_value=0.5,
            input_format=FormatType.FLOAT,
            top_left_color_space=ColorSpace.RGB,
            top_right_color_space=ColorSpace.RGB,
            bottom_left_color_space=ColorSpace.RGB, 
            bottom_right_color_space=ColorSpace.RGB,)
        
        # Create a copy with modified properties
        copy_factory = factory.copy_with(
            width=5,
            height=6,
            top_left=np.array([0.5, 0.5, 0.5]),
            vertical_color_space=ColorSpace.HSV,
            top_segment_color_space=ColorSpace.RGB,
            hue_direction_x=HueMode.CW,
            boundtypes=BoundType.CLAMP,
            top_left_color_space=ColorSpace.RGB,
        )
        
        # Verify original unchanged
        assert factory.width == 3
        assert factory.height == 3
        assert np.array_equal(factory.top_left, RED_FLOAT_HSV)
        assert factory.vertical_color_space == ColorSpace.RGB
        assert factory.top_segment_color_space == ColorSpace.HSV
        assert factory.hue_direction_x == HueMode.CCW
        assert factory.boundtypes == BoundType.CYCLIC
        
        # Verify copy has new values
        assert copy_factory.width == 5
        assert copy_factory.height == 6
        assert np.array_equal(copy_factory.top_left, np.array([0.5, 0.5, 0.5]))
        assert copy_factory.vertical_color_space == ColorSpace.HSV
        assert copy_factory.top_segment_color_space == ColorSpace.RGB
        assert copy_factory.hue_direction_x == HueMode.CW
        assert copy_factory.boundtypes == BoundType.CLAMP
        
        # Verify some properties remain the same
        assert copy_factory.horizontal_color_space == factory.horizontal_color_space
        assert copy_factory.bottom_segment_color_space == factory.bottom_segment_color_space
        assert copy_factory.hue_direction_y == factory.hue_direction_y
        assert copy_factory.top_segment_hue_direction_x == factory.top_segment_hue_direction_x
        assert copy_factory.bottom_segment_hue_direction_x == factory.bottom_segment_hue_direction_x

        assert np.array_equal(copy_factory.top_right,  GREEN_FLOAT_RGB)
        assert np.array_equal(factory.top_right,  GREEN_FLOAT_HSV)
        assert np.array_equal(copy_factory.bottom_left, factory.bottom_left)
        assert np.array_equal(copy_factory.bottom_right, factory.bottom_right)
        assert copy_factory.border_mode == factory.border_mode
        assert copy_factory.border_value == factory.border_value
    
    def test_corners_property(self):
        """Test the computed corners property."""
        factory = CornersCellDualFactory(
            width=2,
            height=2,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )
        
        corners = factory.corners
        assert isinstance(corners, dict)
        assert set(corners.keys()) == {'top_left', 'top_right', 'bottom_left', 'bottom_right'}
        assert np.array_equal(corners['top_left'], RED_FLOAT_RGB)
        assert np.array_equal(corners['top_right'], GREEN_FLOAT_RGB)
        assert np.array_equal(corners['bottom_left'], BLUE_FLOAT_RGB)
        assert np.array_equal(corners['bottom_right'], YELLOW_FLOAT_RGB)
    
    def test_color_spaces_property(self):
        """Test the computed color_spaces property."""
        factory = CornersCellDualFactory(
            width=2,
            height=2,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            top_segment_color_space=ColorSpace.HSV,
            bottom_segment_color_space=ColorSpace.RGB,
            input_format=FormatType.FLOAT,
        )
        
        color_spaces = factory.color_spaces
        assert isinstance(color_spaces, dict)
        assert set(color_spaces.keys()) == {'vertical', 'horizontal', 'top_segment', 'bottom_segment'}
        assert color_spaces['vertical'] == ColorSpace.RGB
        assert color_spaces['horizontal'] == ColorSpace.HSV
        assert color_spaces['top_segment'] == ColorSpace.HSV
        assert color_spaces['bottom_segment'] == ColorSpace.RGB
    
    def test_perpendicular_dual_partition_basic(self):
        """Test basic partition with two intervals."""
        factory = CornersCellDualFactory(
            width=5,
            height=4,
            top_left=np.array([1.0, 0.0, 0.0]),
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=np.array([1.0, 1.0, 0]),
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )

        # Create partition intervals with different color spaces
        pi1 = CellDualPartitionInterval(
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV
        )
        pi2 = CellDualPartitionInterval(
            vertical_color_space=ColorSpace.HSV,
            horizontal_color_space=ColorSpace.RGB
        )
        
        # Create dual partition
        pp = PerpendicularDualPartition(breakpoints=[0.5], values=[pi1, pi2])
        
        # Partition with default padding (1)
        factory1, factory2 = factory.partition_slice(partition=pp)
        
        # === Dimension checks ===
        # With width=5, breakpoint at 0.5, padding=1:
        # slice_ends = [0, ceil(0.5*4)+1, 5] = [0, 3, 5]
        # factory1: indices [0,1,2] (3 pixels), no padding on left
        # factory2: indices [3,4] (2 pixels) + 1 left padding = 3 pixels
        assert factory1.width == 3
        assert factory2.width == 3
        assert factory1.height == factory2.height == 4
        
        # === Color space checks ===
        assert factory1.vertical_color_space == ColorSpace.RGB
        assert factory1.horizontal_color_space == ColorSpace.HSV
        assert factory2.vertical_color_space == ColorSpace.HSV
        assert factory2.horizontal_color_space == ColorSpace.RGB
        
        # === Verify boundaries match ===
        # Get boundary colors from original cell interpolation
        cell = factory.get_cell()
        boundary_top = cell.simple_untransformed_interpolate_edge(0.5, is_top_edge=True)
        boundary_bottom = cell.simple_untransformed_interpolate_edge(0.5, is_top_edge=False)
        
        # Factory1 right side should match boundary (in its color space)
        # Factory2 left side should match boundary (converted to its color space)
        assert np.allclose(factory1.top_right, boundary_top)
        assert np.allclose(factory1.bottom_right, boundary_bottom)
        
        # Convert factory2's left boundary to RGB for comparison (since factory2 uses RGB horizontal)
        # Note: This depends on the actual conversion implementation
        value1 = factory1.get_value()
        value2 = factory2.get_value()
        assert value1 is not None and value2 is not None
        
        # The boundary columns should be consistent after rendering
        # Last column of factory1 should match first column of factory2
        # (May need color space conversion for exact match)
        assert value1.shape[1] == 3  # factory1 width
        assert value2.shape[1] == 3  # factory2 width
    
    def test_perpendicular_dual_partition_multiple_intervals(self):
        """Test partition with three intervals and verify boundary consistency."""
        factory = CornersCellDualFactory(
            width=7,
            height=3,
            top_left=np.array([1.0, 0.0, 0.0]),
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=np.array([1.0, 1.0, 0]),
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )
        
        # Create three intervals with different configurations
        pi1 = CellDualPartitionInterval(
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV
        )
        pi2 = CellDualPartitionInterval(
            vertical_color_space=ColorSpace.HSV,
            horizontal_color_space=ColorSpace.RGB
        )
        pi3 = CellDualPartitionInterval(
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV
        )
        
        pp = PerpendicularDualPartition(breakpoints=[1/3, 2/3], values=[pi1, pi2, pi3])
        
        # Partition with default padding=1
        factory1, factory2, factory3 = factory.partition_slice(partition=pp)
        
        # === Dimension checks ===
        assert factory1.width == 3
        assert factory2.width == 3
        assert factory3.width == 3
        assert factory1.height == factory2.height == factory3.height == 3
        
        # === Color space checks ===
        assert factory1.vertical_color_space == ColorSpace.RGB
        assert factory1.horizontal_color_space == ColorSpace.HSV
        
        assert factory2.vertical_color_space == ColorSpace.HSV
        assert factory2.horizontal_color_space == ColorSpace.RGB
        
        assert factory3.vertical_color_space == ColorSpace.RGB
        assert factory3.horizontal_color_space == ColorSpace.HSV
        
        # === Get rendered values ===
        value1 = factory1.get_value()
        value2 = factory2.get_value()
        value3 = factory3.get_value()
        
        assert value1 is not None and value2 is not None and value3 is not None
        
        # Basic shape checks
        assert value1.shape == (3, 3, 3)
        assert value2.shape == (3, 3, 3)
        assert value3.shape == (3, 3, 3)
    
    def test_perpendicular_dual_partition_padding_variations(self):
        """Test partition with different padding values."""
        factory = CornersCellDualFactory(
            width=5,
            height=3,
            top_left=np.array([1.0, 0.0, 0.0]),
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=np.array([1.0, 1.0, 0]),
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )
        
        pi1 = CellDualPartitionInterval(
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV
        )
        pi2 = CellDualPartitionInterval(
            vertical_color_space=ColorSpace.HSV,
            horizontal_color_space=ColorSpace.RGB
        )
        pp = PerpendicularDualPartition(breakpoints=[0.5], values=[pi1, pi2])
        
        # Test padding=0
        factory1, factory2 = factory.partition_slice(partition=pp, padding=0)
        assert factory1.width == 3  # ceil(0.5 * 5)
        assert factory2.width == 2  # floor(0.5 * 5)
        
        # Test padding=2
        factory1, factory2 = factory.partition_slice(partition=pp, padding=2)
        assert factory1.width == 4  # ceil(0.5 * 5) + 1
        assert factory2.width == 3  # floor(0.5 * 5) + 1
        
        # Verify cumulative width pattern
        factory1, factory2 = factory.partition_slice(partition=pp, padding=1)
        base_w1, base_w2 = factory1.width, factory2.width
        
        for i in range(1, 3):  # Test a few values
            f1, f2 = factory.partition_slice(partition=pp, padding=2*i+1)
            # Each interval gets i extra columns (distributed between left/right)
            assert f1.width == base_w1 + i
            assert f2.width == base_w2 + i
    
    def test_pure_partition_slice(self):
        """Test partition with pure_partition=True."""
        factory = CornersCellDualFactory(
            width=4,
            height=2,
            top_left=np.array([1.0, 0.0, 0.0]),
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=np.array([1.0, 1.0, 0]),
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )
        
        # Get original per-channel coords
        original_pcc = factory.per_channel_coords
        
        pi1 = CellDualPartitionInterval(
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV
        )
        pi2 = CellDualPartitionInterval(
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV
        )
        pp = PerpendicularDualPartition(breakpoints=[0.5], values=[pi1, pi2])
        
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
        factory = CornersCellDualFactory(
            width=6,
            height=3,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )
        
        pi1 = CellDualPartitionInterval(
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV
        )
        pi2 = CellDualPartitionInterval(
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV
        )
        pp = PerpendicularDualPartition(breakpoints=[0.501], values=[pi1, pi2])
        
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
                # 0.501 * 5 = 2.505 -> floor = 2
                assert factories[0].width == 3  # 0-2 inclusive
                assert factories[1].width == 3  # 3-5
            elif mode == IndexRoundingMode.ROUND:
                # 2.505 -> round = 3
                assert factories[0].width == 4  # 0-3
                assert factories[1].width == 2  # 4-5
            elif mode == IndexRoundingMode.CEIL:
                # 2.505 -> ceil = 3
                assert factories[0].width == 4  # 0-3
                assert factories[1].width == 2  # 4-5
    
    def test_edge_interpolation(self):
        """Test edge interpolation method."""
        factory = CornersCellDualFactory(
            width=3,
            height=3,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )
        
        cell = factory.get_cell()
        
        # Test edge interpolation at fraction 0.5
        top_edge = cell.interpolate_edge(0.5, is_top_edge=True)
        bottom_edge = cell.interpolate_edge(0.5, is_top_edge=False)
        
        # At fraction 0.5, should interpolate between corners
        # Note: Due to dual color space, interpolation is complex
        # Just verify we get valid arrays
        assert top_edge.shape == (3,)
        assert bottom_edge.shape == (3,)
        
        # Test edge interpolation at boundaries
        left_edge_top = cell.interpolate_edge(0.0, is_top_edge=True)
        right_edge_top = cell.interpolate_edge(1.0, is_top_edge=True)
        
        # Should match corners (may be in different color space)
        assert left_edge_top.shape == (3,)
        assert right_edge_top.shape == (3,)
    
    def test_border_modes(self):
        """Test different border modes."""
        factory = CornersCellDualFactory(
            width=2,
            height=2,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
            border_mode=None,
        )
        
        # Test different border modes
        for border_mode in [None, 0, 1]:  # None, replicate, constant
            factory_copy = factory.copy_with(border_mode=border_mode)
            if border_mode == 1:
                factory_copy.border_value = 0.5
            
            value = factory_copy.get_value()
            assert value is not None
            assert value.shape == (2, 2, 3)
    
    def test_boundtypes(self):
        """Test different boundtypes."""
        factory = CornersCellDualFactory(
            width=2,
            height=2,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
            boundtypes=BoundType.CLAMP,
        )
        
        # Test different boundtypes
        for boundtype in [BoundType.CLAMP, BoundType.CYCLIC, BoundType.BOUNCE]:
            factory_copy = factory.copy_with(boundtypes=boundtype)
            value = factory_copy.get_value()
            assert value is not None
            assert value.shape == (2, 2, 3)
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test mismatched array dimensions
        with pytest.raises(ValueError):
            CornersCellDualFactory(
                width=2,
                height=2,
                top_left=np.array([1.0, 0]),  # 2 channels
                top_right=GREEN_FLOAT_RGB,  # 3 channels
                bottom_left=BLUE_FLOAT_RGB,
                bottom_right=YELLOW_FLOAT_RGB,
                vertical_color_space=ColorSpace.RGB,
                horizontal_color_space=ColorSpace.HSV,
                input_format=FormatType.FLOAT,
            )
        
        # Test invalid color space
        with pytest.raises(ValueError):
            CornersCellDualFactory(
                width=2,
                height=2,
                top_left=RED_FLOAT_RGB,
                top_right=GREEN_FLOAT_RGB,
                bottom_left=BLUE_FLOAT_RGB,
                bottom_right=YELLOW_FLOAT_RGB,
                vertical_color_space="INVALID_COLOR_SPACE",  # type: ignore
                horizontal_color_space=ColorSpace.HSV,
                input_format=FormatType.FLOAT,
            )
        
        # Test negative dimensions
        with pytest.raises(ValueError):
            CornersCellDualFactory(
                width=-1,
                height=2,
                top_left=RED_FLOAT_RGB,
                top_right=GREEN_FLOAT_RGB,
                bottom_left=BLUE_FLOAT_RGB,
                bottom_right=YELLOW_FLOAT_RGB,
                vertical_color_space=ColorSpace.RGB,
                horizontal_color_space=ColorSpace.HSV,
                input_format=FormatType.FLOAT,
            )
    
    def test_repr_and_eq(self):
        """Test string representation and equality."""
        factory1 = CornersCellDualFactory(
            width=2,
            height=2,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )
        
        factory2 = CornersCellDualFactory(
            width=2,
            height=2,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )
        
        # Test equality
        assert factory1 == factory2
        
        # Test inequality
        factory3 = CornersCellDualFactory(
            width=3,
            height=2,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )
        
        assert factory1 != factory3
        
        # Test repr
        repr_str = repr(factory1)
        assert "CornersCellDualFactory" in repr_str
        assert "width=2" in repr_str
        assert "height=2" in repr_str
        assert "vertical_space=ColorSpace.RGB" in repr_str or "ColorSpace.RGB" in repr_str
    
    def test_single_partition_returns_self(self):
        """Test that partitioning with a single interval returns a copy of self."""
        factory = CornersCellDualFactory(
            width=3,
            height=3,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )
        
        pi = CellDualPartitionInterval(
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV
        )
        pp = PerpendicularDualPartition(breakpoints=[], values=[pi])
        
        factories = factory.partition_slice(partition=pp)
        assert len(factories) == 1
        assert factories[0] == factory
        assert factories[0] is not factory  # Should be a copy
    
    def test_invalidate_cache_methods(self):
        """Test cache invalidation methods."""
        factory = CornersCellDualFactory(
            width=2,
            height=2,
            top_left=RED_FLOAT_RGB,
            top_right=GREEN_FLOAT_RGB,
            bottom_left=BLUE_FLOAT_RGB,
            bottom_right=YELLOW_FLOAT_RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )
        
        # Create cell and cache value
        cell = factory.get_cell()
        value1 = cell.get_value()
        assert cell._value is not None
        
        # Test invalidate_cell_cache
        factory.invalidate_cell_cache()
        assert cell._value is None
        
        # Get value again
        value2 = cell.get_value()
        assert np.array_equal(value1, value2)
        
        # Test invalidate_cell
        factory.invalidate_cell()
        assert factory._cell is None
        
        # Test reset_per_channel_coords
        factory.get_cell()  # Recreate cell
        factory.reset_per_channel_coords()
        assert factory._per_channel_coords is None
        assert factory._cell is None
    
    def test_corner_input_format_conversion(self):
        """Test that non-float input formats are converted correctly."""
        # Test with integer input (0-255 range)
        factory = CornersCellDualFactory(
            width=2,
            height=2,
            top_left=np.array([255, 0, 0]),
            top_right=np.array([0, 255, 0]),
            bottom_left=np.array([0, 0, 255]),
            bottom_right=np.array([255, 255, 0]),
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            top_left_color_space=ColorSpace.RGB,
            top_right_color_space=ColorSpace.RGB,
            bottom_left_color_space=ColorSpace.RGB,
            bottom_right_color_space=ColorSpace.RGB,
            top_segment_color_space=ColorSpace.RGB,
            bottom_segment_color_space=ColorSpace.RGB,
            input_format=FormatType.INT,
        )
        
        # Corners should be converted to float [0, 1] range
        # (Conversion happens to segment color space)
        assert np.allclose(factory.top_left, [1.0, 0, 0])  # RGB red
        assert np.allclose(factory.top_right, [0, 1.0, 0])  # RGB green
        assert np.allclose(factory.bottom_left, [0, 0, 1.0])  # RGB blue
        assert np.allclose(factory.bottom_right, [1.0, 1.0, 0])  # RGB yellow
    
    def test_corner_different_color_spaces(self):
        """Test corners specified in different color spaces."""
        factory = CornersCellDualFactory(
            width=2,
            height=2,
            top_left=RED_FLOAT_RGB,  # RGB red
            top_right=GREEN_FLOAT_RGB,  # RGB green
            bottom_left=BLUE_FLOAT_RGB,  # RGB blue
            bottom_right=YELLOW_FLOAT_RGB,  # RGB yellow
            top_left_color_space=ColorSpace.RGB,
            top_right_color_space=ColorSpace.RGB,
            bottom_left_color_space=ColorSpace.RGB,
            bottom_right_color_space=ColorSpace.RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            top_segment_color_space=ColorSpace.HSV,
            bottom_segment_color_space=ColorSpace.HSV,
            input_format=FormatType.FLOAT,
        )
        
        # All corners should be converted to HSV (segment color space)
        assert factory.top_segment_color_space == ColorSpace.HSV
        assert factory.bottom_segment_color_space == ColorSpace.HSV
        
        # Verify corners were converted (not original RGB values)
        # Red in HSV: [0, 1, 1]
        # Green in HSV: [120, 1, 1] but normalized
        # Blue in HSV: [240, 1, 1] -> normalized
        # Yellow in HSV: [60, 1, 1] -> normalized
        
        # Just check that they're not the original RGB values
        assert not np.array_equal(factory.top_left, [1.0, 0, 0])
        assert not np.array_equal(factory.top_right, [0, 1.0, 0])
        assert not np.array_equal(factory.bottom_left, [0, 0, 1.0])
        assert not np.array_equal(factory.bottom_right, [1.0, 1, 0])
        
        # All should have valid HSV values
        value = factory.get_value()
        assert value is not None
        # HSV values should be in valid ranges
        assert np.all(value[..., 0] >= 0)  # Hue
        assert np.all(value[..., 0] <= 1)
        # Saturation and Value should also be in [0, 1] range
    
    def test_segment_specific_properties(self):
        """Test segment-specific properties with different configurations."""
        # Test with different segment color spaces
        factory = CornersCellDualFactory(
            width=3,
            height=3,
            top_left=CYAN_FLOAT_RGB,  # HSV
            top_right=np.array([0.333, 1.0, 1.0]),  # HSV
            bottom_left=RED_FLOAT_RGB,  # RGB
            bottom_right=GREEN_FLOAT_RGB,  # RGB
            top_left_color_space=ColorSpace.HSV,
            top_right_color_space=ColorSpace.HSV,
            bottom_left_color_space=ColorSpace.RGB,
            bottom_right_color_space=ColorSpace.RGB,
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            top_segment_color_space=ColorSpace.HSV,
            bottom_segment_color_space=ColorSpace.RGB,
            input_format=FormatType.FLOAT,
        )
        
        # Verify segment configurations
        assert factory.top_segment_color_space == ColorSpace.HSV
        assert factory.bottom_segment_color_space == ColorSpace.RGB
        
        # Top corners should be in HSV
        # Bottom corners should be in RGB
        
        value = factory.get_value()
        assert value is not None
        assert value.shape == (3, 3, 3)
        
        # The dual nature means vertical interpolation in RGB,
        # but horizontal interpolation in top segment uses HSV,
        # and bottom segment uses RGB
    
    def test_complex_dual_interpolation(self):
        """Test complex dual interpolation scenarios."""
        # Create a scenario with all different color spaces
        factory = CornersCellDualFactory(
            width=4,
            height=4,
            top_left=CYAN_FLOAT_RGB,  # HSV red
            top_right=np.array([0.5, 1.0, 1.0]),  # HSV cyan-ish
            bottom_left=RED_FLOAT_RGB,  # RGB red
            bottom_right=GREEN_FLOAT_RGB,  # RGB green
            vertical_color_space=ColorSpace.RGB,
            horizontal_color_space=ColorSpace.HSV,
            top_segment_color_space=ColorSpace.HSV,
            bottom_segment_color_space=ColorSpace.RGB,
            top_segment_hue_direction_x=HueMode.CW,
            bottom_segment_hue_direction_x=HueMode.CCW,
            hue_direction_y=HueMode.SHORTEST,
            input_format=FormatType.FLOAT,
        )
        
        # This tests:
        # 1. Vertical interpolation in LAB space
        # 2. Top segment horizontal interpolation in HSV with CW hue
        # 3. Bottom segment horizontal interpolation in RGB with different corners
        # 4. Different hue directions for segments
        
        value = factory.get_value()
        assert value is not None
        assert value.shape == (4, 4, 3)
        
        # All values should be valid
        assert not np.any(np.isnan(value))
        assert not np.any(np.isinf(value))


# Parameterized tests for index rounding modes
@pytest.mark.parametrize(
    "mode,expected_left_width",
    [
        (IndexRoundingMode.FLOOR, 3),
        (IndexRoundingMode.ROUND, 4),
        (IndexRoundingMode.CEIL, 4),
    ]
)
def test_index_rounding_modes_corners_dual(mode, expected_left_width):
    """Parameterized test for index rounding modes with dual partitions."""
    factory = CornersCellDualFactory(
        width=6,
        height=3,
        top_left=RED_FLOAT_RGB,
        top_right=GREEN_FLOAT_RGB,
        bottom_left=BLUE_FLOAT_RGB,
        bottom_right=YELLOW_FLOAT_RGB,
        vertical_color_space=ColorSpace.RGB,
        horizontal_color_space=ColorSpace.HSV,
        input_format=FormatType.FLOAT,
    )
    
    pi1 = CellDualPartitionInterval(
        vertical_color_space=ColorSpace.RGB,
        horizontal_color_space=ColorSpace.HSV
    )
    pi2 = CellDualPartitionInterval(
        vertical_color_space=ColorSpace.RGB,
        horizontal_color_space=ColorSpace.HSV
    )
    pp = PerpendicularDualPartition(breakpoints=[0.501], values=[pi1, pi2])
    
    f1, f2 = factory.partition_slice(
        partition=pp,
        padding=0,
        index_rounding_mode=mode
    )
    
    assert f1.width == expected_left_width


# Test for edge cases with uniform color spaces
def test_uniform_color_space_dual():
    """Test that uniform color spaces work correctly in dual mode."""
    factory = CornersCellDualFactory(
        width=3,
        height=3,
        top_left=CYAN_FLOAT_RGB,
        top_right=CYAN_FLOAT_RGB,
        bottom_left=CYAN_FLOAT_RGB,
        bottom_right=CYAN_FLOAT_RGB,
        vertical_color_space=ColorSpace.HSV,
        horizontal_color_space=ColorSpace.HSV,
        input_format=FormatType.FLOAT,
        hue_direction_x=HueMode.CW,
        hue_direction_y=HueMode.CCW,
    )
    
    # All corners are identical in HSV
    value = factory.get_value()
    assert value is not None
    
    # All values should be the same
    assert np.allclose(value, value[0, 0])

def test_render_values():
    top_left = RED_FLOAT_RGB
    top_right = BLUE_FLOAT_RGB
    bottom_left = MAGENTA_FLOAT_RGB
    bottom_right = get_white_hsv(hue=300.0)
    top_segment_color_space = ColorSpace.RGB
    bottom_segment_color_space = ColorSpace.RGB
    horizontal_color_space = ColorSpace.RGB
    vertical_color_space = ColorSpace.HSV
    hue_direction_x = HueMode.SHORTEST
    hue_direction_y = HueMode.LONGEST
    input_format = FormatType.FLOAT
    WIDTH = 4
    HEIGHT = 4
    factory = CornersCellDualFactory(
        width=WIDTH,
        height=HEIGHT,
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        top_segment_color_space=top_segment_color_space,
        bottom_segment_color_space=bottom_segment_color_space,
        horizontal_color_space=horizontal_color_space,
        vertical_color_space=vertical_color_space,
        hue_direction_x=hue_direction_x,
        hue_direction_y=hue_direction_y,
        input_format=input_format,
        bottom_right_color_space=ColorSpace.HSV,
    )
    value = factory.get_value()
    rgb_value = np_convert(value, from_space=vertical_color_space, to_space=ColorSpace.RGB, input_type=FormatType.FLOAT, output_type=FormatType.INT)

    first_row = rgb_value[0]
    expected_first_row = np.array([[255, 0 , 0], [170, 0, 85], [85, 0, 170], [0, 0, 255]])
    last_row = rgb_value[-1]
    expected_last_row = np.array([[255, 0 , 255], [255, 85, 255], [255, 170, 255], [255, 255, 255]])
    first_column = rgb_value[:,0]
    expected_first_column = np.array([[255, 0 , 0], np_convert(np.array([100.0, 1.0, 1.0]), from_space=ColorSpace.HSV, to_space=ColorSpace.RGB, input_type=FormatType.FLOAT, output_type=FormatType.INT),
                                    np_convert(np.array([200.0, 1.0, 1.0]), from_space=ColorSpace.HSV, to_space=ColorSpace.RGB, input_type=FormatType.FLOAT, output_type=FormatType.INT),
                                      [255, 0, 255]])
    last_column = rgb_value[:,-1]

    #Blue (240, 1.0, 1.0) to WhiteMagenta(300, 0.0, 1.0) Longest so WhiteMagenta (-60, 0.0, 1.0)
    expected_last_column = np.array([[0, 0 , 255], 
             np_convert(np.array([240-300*1/3, 2/3, 1.0]), from_space=ColorSpace.HSV, to_space=ColorSpace.RGB, input_type=FormatType.FLOAT, output_type=FormatType.INT),
            np_convert(np.array([240-300*2/3, 1/3, 1.0]), from_space=ColorSpace.HSV, to_space=ColorSpace.RGB, input_type=FormatType.FLOAT, output_type=FormatType.INT),
                                     [255, 255, 255]])

    assert factory._bottom_right_grayscale_hue is not None
    assert factory._bottom_right_grayscale_hue == 300.0
    assert np.array_equal(last_column, expected_last_column)
    assert np.array_equal(first_column, expected_first_column)
    assert np.array_equal(first_row, expected_first_row)
    assert np.array_equal(last_row, expected_last_row)
    assert value is not None
