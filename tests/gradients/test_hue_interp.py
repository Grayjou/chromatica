from ...examples.animations import filling_ring_problematic

def test_filling_ring_problematic():
    #frames after which issues arise
    total_hue_increases = list(range(70, 121, 10))
    for hue_increase in total_hue_increases:
        print(f"Testing filling_ring_problematic with total_hue_increase = {hue_increase}")
        frames_with_issues = filling_ring_problematic(total_hue_increase=hue_increase)
        assert not frames_with_issues, f"Issues detected in frames: {frames_with_issues}"