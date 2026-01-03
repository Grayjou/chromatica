from __future__ import annotations
import numpy as np
import pytest
from typing import cast, Optional

from ....chromatica.gradients.gradient2dv2.generators.cell_corners import CornersCellFactory
from ....chromatica.gradients.gradient2dv2.cell.corners import CornersCell
from ....chromatica.gradients.gradient2dv2.partitions import PerpendicularPartition, PartitionInterval, IndexRoundingMode
from ....chromatica.types.color_types import ColorMode, HueDirection
from ....chromatica.types.format_type import FormatType
from ....chromatica.conversions import np_convert
from boundednumbers import BoundType
from unitfield import upbm_2d


class TestCornersCellFactory:
    def test_init(self):
        """Test basic initialization and lazy cell creation."""
        factory = CornersCellFactory(
            width=3,
            height=3,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
            hue_direction_x='shortest',
            hue_direction_y='shortest',
            boundtypes=BoundType.CLAMP,
            border_mode=None,
            border_value=None,
        )
        
        assert isinstance(factory, CornersCellFactory)
        assert factory._cell is None  # Cell is initially None
        
        # Get cell (lazy creation)
        cell = factory.get_cell()
        assert isinstance(cell, CornersCell)
        assert factory._cell is cell  # Cell is now set
        
        # Check properties are synced
        assert np.array_equal(factory.top_left, np.array([1.0, 0, 0]))
        assert np.array_equal(factory.top_right, np.array([0, 1.0, 0]))
        assert np.array_equal(factory.bottom_left, np.array([0, 0, 1.0]))
        assert np.array_equal(factory.bottom_right, np.array([1.0, 1, 0]))
        assert factory.width == 3
        assert factory.height == 3
        assert factory.color_mode == ColorMode.RGB
        assert factory.hue_direction_x == 'shortest'
        assert factory.hue_direction_y == 'shortest'
    
    def test_property_synchronization(self):
        """Test that property changes sync with the underlying cell."""
        factory = CornersCellFactory(
            width=3,
            height=3,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
        )
        
        # Create cell first
        cell = factory.get_cell()
        
        # Change properties and verify sync
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
        factory.hue_direction_x = HueDirection.CCW
        assert cell.hue_direction_x == HueDirection.CCW
        
        factory.hue_direction_y = HueDirection.CW
        assert cell.hue_direction_y == HueDirection.CW
    
    def test_get_value(self):
        """Test value retrieval with and without cell initialization."""
        factory = CornersCellFactory(
            width=3,
            height=3,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
        )
        
        # Test without initialization (init_cell=False)
        value = factory.get_value(init_cell=False)
        assert value is None  # No cell created yet
        
        # Test with initialization
        value = factory.get_value(init_cell=True)
        assert value is not None
        assert value.shape == (3, 3, 3)  # height, width, channels
        
        # Test specific values (check corners)
        assert np.allclose(value[0, 0], [1.0, 0, 0])      # top-left
        assert np.allclose(value[0, -1], [0, 1.0, 0])     # top-right
        assert np.allclose(value[-1, 0], [0, 0, 1.0])     # bottom-left
        assert np.allclose(value[-1, -1], [1.0, 1, 0])    # bottom-right
    
    def test_get_value_auto_init(self):
        """Test get_value with default init_cell=True."""
        factory = CornersCellFactory(
            width=2,
            height=2,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
        )
        
        # Default behavior should create cell
        value = factory.get_value()
        assert value is not None
        assert value.shape == (2, 2, 3)
    
    def test_per_channel_coords(self):
        """Test per-channel coordinates handling."""
        factory = CornersCellFactory(
            width=2,
            height=2,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
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
        factory = CornersCellFactory(
            width=2,
            height=2,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
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
        factory2 = CornersCellFactory(
            width=2,
            height=2,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
        )
        value2 = factory2.get_value()
        
        # Values should be different due to transforms
        assert not np.allclose(value, value2)
    
    def test_dimension_changes(self):
        """Test that width/height changes invalidate the cell."""
        factory = CornersCellFactory(
            width=2,
            height=2,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
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
    
    def test_color_mode_conversion(self):
        """Test color space conversion."""
        # Start with RGB colors
        factory = CornersCellFactory(
            width=2,
            height=2,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
        )
        
        # Convert to HSV
        factory.color_mode = ColorMode.HSV
        
        # Verify conversion happened
        assert factory.color_mode == ColorMode.HSV
        factory.hue_direction_x = HueDirection.CCW
        factory.hue_direction_y = HueDirection.CW
        # Get value in HSV space
        value = factory.get_value()
        assert value is not None
        assert value.shape == (2, 2, 3)
        
        # Verify corners were converted
        # Convert original corners to HSV for comparison
        hsv_top_left = np_convert(
            np.array([1.0, 0, 0]), 
            from_space=ColorMode.RGB,
            to_space=ColorMode.HSV,
            input_type='float',
            output_type='float'
        )
        assert np.allclose(factory.top_left, hsv_top_left)
    
    def test_copy_with(self):
        """Test the copy_with method."""
        factory = CornersCellFactory(
            width=3,
            height=3,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
            hue_direction_x=HueDirection.CCW,
            hue_direction_y=HueDirection.CW,
            boundtypes=BoundType.CYCLIC,
            border_mode=1,
            border_value=0.5,
        )
        
        # Create a copy with modified properties
        copy_factory = factory.copy_with(
            width=5,
            height=6,
            top_left=np.array([0.5, 0.5, 0.5]),
            hue_direction_x=HueDirection.CW,
            boundtypes=BoundType.CLAMP,
        )
        
        # Verify original unchanged
        assert factory.width == 3
        assert factory.height == 3
        assert np.array_equal(factory.top_left, np.array([1.0, 0, 0]))
        assert factory.hue_direction_x == HueDirection.CCW
        assert factory.boundtypes == BoundType.CYCLIC
        
        # Verify copy has new values
        assert copy_factory.width == 5
        assert copy_factory.height == 6
        assert np.array_equal(copy_factory.top_left, np.array([0.5, 0.5, 0.5]))
        assert copy_factory.hue_direction_x == HueDirection.CW
        assert copy_factory.boundtypes == BoundType.CLAMP
        
        # Verify some properties remain the same
        assert copy_factory.color_mode == factory.color_mode
        assert copy_factory.hue_direction_y == factory.hue_direction_y
        assert np.array_equal(copy_factory.top_right, factory.top_right)
        assert np.array_equal(copy_factory.bottom_left, factory.bottom_left)
        assert np.array_equal(copy_factory.bottom_right, factory.bottom_right)
        assert copy_factory.border_mode == factory.border_mode
        assert copy_factory.border_value == factory.border_value
    
    def test_corners_property(self):
        """Test the computed corners property."""
        factory = CornersCellFactory(
            width=2,
            height=2,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
        )
        
        corners = factory.corners
        assert isinstance(corners, dict)
        assert set(corners.keys()) == {'top_left', 'top_right', 'bottom_left', 'bottom_right'}
        assert np.array_equal(corners['top_left'], np.array([1.0, 0, 0]))
        assert np.array_equal(corners['top_right'], np.array([0, 1.0, 0]))
        assert np.array_equal(corners['bottom_left'], np.array([0, 0, 1.0]))
        assert np.array_equal(corners['bottom_right'], np.array([1.0, 1, 0]))
    
    def test_perpendicular_partition_basic(self):
        """Test basic partition with two intervals."""
        factory = CornersCellFactory(
            width=5,
            height=4,
            top_left=np.array([1.0, 0.0, 0.0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1.0, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
        )

        pi1 = PartitionInterval("rgb")
        pi2 = PartitionInterval("hsv", hue_direction_x=HueDirection.CCW, hue_direction_y=HueDirection.CW)
        pp = PerpendicularPartition(breakpoints=[0.5], values=[pi1, pi2])
        
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
        assert factory1.color_mode == ColorMode.RGB
        assert factory2.color_mode == ColorMode.HSV
        
        # === Factory1 corner checks (RGB, no conversion needed) ===
        # Factory1 gets left side, corners should match original left side
        assert np.allclose(factory1.top_left, factory.top_left)
        assert np.allclose(factory1.bottom_left, factory.bottom_left)
        
        # === Factory2 corner checks (HSV, need conversion) ===
        # Factory2 gets right side converted to HSV
        # Convert back to RGB for comparison
        factory2_top_left_rgb = np_convert(
            factory2.top_left, 
            from_space=ColorMode.HSV, 
            to_space=ColorMode.RGB,
            input_type='float', 
            output_type='float'
        )
        factory2_bottom_left_rgb = np_convert(
            factory2.bottom_left,
            from_space=ColorMode.HSV,
            to_space=ColorMode.RGB,
            input_type='float',
            output_type='float'
        )
        
        # Factory2's left side should match factory1's right side (shared boundary)
        # Get boundary colors from cell interpolation
        cell = factory.get_cell()
        boundary_top = cell.simple_untransformed_interpolate_edge(0.5, is_top_edge=True)
        boundary_bottom = cell.simple_untransformed_interpolate_edge(0.5, is_top_edge=False)
        
        assert np.allclose(factory1.top_right, boundary_top)
        assert np.allclose(factory1.bottom_right, boundary_bottom)
        #assert np.allclose(factory2_top_left_rgb, boundary_top)
        assert np.allclose(factory2_bottom_left_rgb, boundary_bottom)
        
        # Factory2's right side should match original right corners (converted)
        factory2_top_right_rgb = np_convert(
            factory2.top_right,
            from_space=ColorMode.HSV,
            to_space=ColorMode.RGB,
            input_type='float',
            output_type='float'
        )
        factory2_bottom_right_rgb = np_convert(
            factory2.bottom_right,
            from_space=ColorMode.HSV,
            to_space=ColorMode.RGB,
            input_type='float',
            output_type='float'
        )
        
        assert np.allclose(factory2_top_right_rgb, factory.top_right)
        assert np.allclose(factory2_bottom_right_rgb, factory.bottom_right)
    
    def test_perpendicular_partition_multiple_intervals(self):
        """Test partition with three intervals and verify boundary consistency."""
        factory = CornersCellFactory(
            width=7,
            height=3,
            top_left=np.array([1.0, 0.0, 0.0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1.0, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
        )
        
        pi1 = PartitionInterval("rgb")
        pi2 = PartitionInterval("hsv", hue_direction_x=HueDirection.CCW, hue_direction_y=HueDirection.CW)
        pi3 = PartitionInterval("rgb")
        pp = PerpendicularPartition(breakpoints=[1/3, 2/3], values=[pi1, pi2, pi3])
        
        # Partition with default padding=1
        factory1, factory2, factory3 = factory.partition_slice(partition=pp)
        
        # === Dimension checks ===
        # With width=7, breakpoints at 1/3 and 2/3, padding=1:
        # slice_ends = [0, ceil(1/3*6)+1, ceil(2/3*6)+1, 7] ≈ [0, 3, 5, 7]
        # Widths after padding: [3, 3, 3]
        assert factory1.width == 3
        assert factory2.width == 3
        assert factory3.width == 3
        assert factory1.height == factory2.height == factory3.height == 3
        
        # === Color space checks ===
        assert factory1.color_mode == ColorMode.RGB
        assert factory2.color_mode == ColorMode.HSV
        assert factory3.color_mode == ColorMode.RGB
        
        # === Boundary consistency checks ===
        # Get original cell for boundary interpolation
        cell = factory.get_cell()
        
        # Boundary between factory1 and factory2 at 1/3
        boundary1_top = cell.simple_untransformed_interpolate_edge(1/3, is_top_edge=True)
        boundary1_bottom = cell.simple_untransformed_interpolate_edge(1/3, is_top_edge=False)

        # Factory1 right side should match boundary
        assert np.allclose(factory1.top_right, boundary1_top)
        assert np.allclose(factory1.bottom_right, boundary1_bottom)
        
        # Factory2 left side should match boundary (converted from HSV)
        factory2_top_left_rgb = np_convert(
            factory2.top_left,
            from_space=ColorMode.HSV,
            to_space=ColorMode.RGB,
            input_type='float',
            output_type='float'
        )
        factory2_bottom_left_rgb = np_convert(
            factory2.bottom_left,
            from_space=ColorMode.HSV,
            to_space=ColorMode.RGB,
            input_type='float',
            output_type='float'
        )
        
        assert np.allclose(factory2_top_left_rgb, boundary1_top)
        assert np.allclose(factory2_bottom_left_rgb, boundary1_bottom)
        
        # Boundary between factory2 and factory3 at 2/3
        boundary2_top = cell.simple_untransformed_interpolate_edge(2/3, is_top_edge=True)
        boundary2_bottom = cell.simple_untransformed_interpolate_edge(2/3, is_top_edge=False)
        
        # Factory2 right side should match boundary (converted from HSV)
        factory2_top_right_rgb = np_convert(
            factory2.top_right,
            from_space=ColorMode.HSV,
            to_space=ColorMode.RGB,
            input_type='float',
            output_type='float'
        )
        factory2_bottom_right_rgb = np_convert(
            factory2.bottom_right,
            from_space=ColorMode.HSV,
            to_space=ColorMode.RGB,
            input_type='float',
            output_type='float'
        )
        
        assert np.allclose(factory2_top_right_rgb, boundary2_top)
        assert np.allclose(factory2_bottom_right_rgb, boundary2_bottom)
        
        # Factory3 left side should match boundary
        assert np.allclose(factory3.top_left, boundary2_top)
        assert np.allclose(factory3.bottom_left, boundary2_bottom)
    
    def test_perpendicular_partition_padding_variations(self):
        """Test partition with different padding values."""
        factory = CornersCellFactory(
            width=5,
            height=3,
            top_left=np.array([1.0, 0.0, 0.0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1.0, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
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
        factory = CornersCellFactory(
            width=4,
            height=2,
            top_left=np.array([1.0, 0.0, 0.0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1.0, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
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
        factory = CornersCellFactory(
            width=6,
            height=3,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
        )
        
        pi1 = PartitionInterval("rgb")
        pi2 = PartitionInterval("rgb")
        pp = PerpendicularPartition(breakpoints=[0.501], values=[pi1, pi2])
        
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
        factory = CornersCellFactory(
            width=3,
            height=3,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
        )
        
        cell = factory.get_cell()
        
        # Test edge interpolation at fraction 0.5
        top_edge = cell.interpolate_edge(0.5, is_top_edge=True)
        bottom_edge = cell.interpolate_edge(0.5, is_top_edge=False)
        
        # At fraction 0.5, should interpolate between corners
        expected_top = (factory.top_left + factory.top_right) / 2
        expected_bottom = (factory.bottom_left + factory.bottom_right) / 2
        
        assert np.allclose(top_edge, expected_top)
        assert np.allclose(bottom_edge, expected_bottom)
        
        # Test edge interpolation at boundaries
        left_edge_top = cell.interpolate_edge(0.0, is_top_edge=True)
        right_edge_top = cell.interpolate_edge(1.0, is_top_edge=True)
        
        assert np.allclose(left_edge_top, factory.top_left)
        assert np.allclose(right_edge_top, factory.top_right)
    
    def test_border_modes(self):
        """Test different border modes."""
        factory = CornersCellFactory(
            width=2,
            height=2,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
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
        factory = CornersCellFactory(
            width=2,
            height=2,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
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
            CornersCellFactory(
                width=2,
                height=2,
                top_left=np.array([1.0, 0]),  # 2 channels
                top_right=np.array([0, 1.0, 0]),  # 3 channels
                bottom_left=np.array([0, 0, 1.0]),
                bottom_right=np.array([1.0, 1, 0]),
                input_format=FormatType.FLOAT,
                color_mode=ColorMode.RGB,
            )
        
        # Test invalid color space
        with pytest.raises(ValueError):
            CornersCellFactory(
                width=2,
                height=2,
                top_left=np.array([1.0, 0, 0]),
                top_right=np.array([0, 1.0, 0]),
                bottom_left=np.array([0, 0, 1.0]),
                bottom_right=np.array([1.0, 1, 0]),
                input_format=FormatType.FLOAT,
                color_mode="INVALID_color_mode",  # type: ignore
            )
        
        # Test negative dimensions
        with pytest.raises(ValueError):
            CornersCellFactory(
                width=-1,
                height=2,
                top_left=np.array([1.0, 0, 0]),
                top_right=np.array([0, 1.0, 0]),
                bottom_left=np.array([0, 0, 1.0]),
                bottom_right=np.array([1.0, 1, 0]),
                input_format=FormatType.FLOAT,
                color_mode=ColorMode.RGB,
            )
    
    def test_repr_and_eq(self):
        """Test string representation and equality."""
        factory1 = CornersCellFactory(
            width=2,
            height=2,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
        )
        
        factory2 = CornersCellFactory(
            width=2,
            height=2,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
        )
        
        # Test equality
        assert factory1 == factory2
        
        # Test inequality
        factory3 = CornersCellFactory(
            width=3,
            height=2,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
        )
        
        assert factory1 != factory3
        
        # Test repr
        repr_str = repr(factory1)
        assert "CornersCellFactory" in repr_str
        assert "width=2" in repr_str
        assert "height=2" in repr_str
        assert "color_mode=<ColorMode.RGB" in repr_str
    
    def test_hue_space_uniform_dimension_does_not_break(self):
        """Test that uniform hue dimensions don't break interpolation."""
        factory = CornersCellFactory(
            width=3,
            height=3,
            top_left=np.array([0.0, 1.0, 1.0]),   # HSV-ish
            top_right=np.array([0.0, 1.0, 1.0]),
            bottom_left=np.array([0.0, 1.0, 1.0]),
            bottom_right=np.array([0.0, 1.0, 1.0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.HSV,
            hue_direction_x=HueDirection.CW,
            hue_direction_y=HueDirection.CW,
        )
        
        value = factory.get_value()
        
        # All values should be the same since all corners are identical
        assert np.allclose(value[..., 0], 0.0)  # Hue channel
        assert np.allclose(value[..., 1], 1.0)  # Saturation channel
        assert np.allclose(value[..., 2], 1.0)  # Value channel
    
    def test_single_partition_returns_self(self):
        """Test that partitioning with a single interval returns a copy of self."""
        factory = CornersCellFactory(
            width=3,
            height=3,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
        )
        
        pi = PartitionInterval("rgb")
        pp = PerpendicularPartition(breakpoints=[], values=[pi])
        
        factories = factory.partition_slice(partition=pp)
        assert len(factories) == 1
        assert factories[0] == factory
        assert factories[0] is not factory  # Should be a copy
    
    def test_invalidate_cache_methods(self):
        """Test cache invalidation methods."""
        factory = CornersCellFactory(
            width=2,
            height=2,
            top_left=np.array([1.0, 0, 0]),
            top_right=np.array([0, 1.0, 0]),
            bottom_left=np.array([0, 0, 1.0]),
            bottom_right=np.array([1.0, 1, 0]),
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.RGB,
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
        factory = CornersCellFactory(
            width=2,
            height=2,
            top_left=np.array([255, 0, 0]),
            top_right=np.array([0, 255, 0]),
            bottom_left=np.array([0, 0, 255]),
            bottom_right=np.array([255, 255, 0]),
            input_format=FormatType.INT,
            color_mode=ColorMode.RGB,
        )
        
        # Corners should be converted to float [0, 1] range
        assert np.allclose(factory.top_left, [1.0, 0, 0])
        assert np.allclose(factory.top_right, [0, 1.0, 0])
        assert np.allclose(factory.bottom_left, [0, 0, 1.0])
        assert np.allclose(factory.bottom_right, [1.0, 1.0, 0])
    
    def test_corner_different_color_modes(self):
        """Test corners specified in different color spaces."""
        factory = CornersCellFactory(
            width=2,
            height=2,
            top_left=np.array([1.0, 0, 0]),  # RGB red
            top_right=np.array([0, 1.0, 0]),  # RGB green
            bottom_left=np.array([0, 0, 1.0]),  # RGB blue
            bottom_right=np.array([1.0, 1, 0]),  # RGB yellow
            top_left_color_mode=ColorMode.RGB,
            top_right_color_mode=ColorMode.RGB,
            bottom_left_color_mode=ColorMode.RGB,
            bottom_right_color_mode=ColorMode.RGB,
            input_format=FormatType.FLOAT,
            color_mode=ColorMode.HSV,  # Target space
            hue_direction_x=HueDirection.CCW,
            hue_direction_y=HueDirection.CW,
        )
        
        # All corners should be converted to HSV
        assert factory.color_mode == ColorMode.HSV
        
        # Verify conversion happened (hue values should be different)
        # Red in HSV: [0, 1, 1]
        # Green in HSV: [120, 1, 1] but normalized to [0.333, 1, 1] if in [0,1]
        # Blue in HSV: [240, 1, 1] -> [0.667, 1, 1]
        # Yellow in HSV: [60, 1, 1] -> [0.167, 1, 1]
        
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
        assert np.all(value[..., 0] <= 360)
        assert np.all(value[..., 1] >= 0)  # Saturation
        assert np.all(value[..., 1] <= 1)
        assert np.all(value[..., 2] >= 0)  # Value
        assert np.all(value[..., 2] <= 1)


# Parameterized tests
@pytest.mark.parametrize(
    "mode,expected_left_width",
    [
        (IndexRoundingMode.FLOOR, 3),
        (IndexRoundingMode.ROUND, 4),
        (IndexRoundingMode.CEIL, 4),
    ]
)
def test_index_rounding_modes_corners(mode, expected_left_width):
    """Parameterized test for index rounding modes."""
    factory = CornersCellFactory(
        width=6,
        height=3,
        top_left=np.array([1.0, 0, 0]),
        top_right=np.array([0, 1.0, 0]),
        bottom_left=np.array([0, 0, 1.0]),
        bottom_right=np.array([1.0, 1, 0]),
        input_format=FormatType.FLOAT,
        color_mode=ColorMode.RGB,
    )
    
    pp = PerpendicularPartition(
        breakpoints=[0.501],
        values=[PartitionInterval("rgb"), PartitionInterval("rgb")]
    )
    
    f1, f2 = factory.partition_slice(
        partition=pp,
        padding=0,
        index_rounding_mode=mode
    )
    
    assert f1.width == expected_left_width