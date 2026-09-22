from scripts.recovery.native_navigation_gate import run_gate


def test_native_navigation_cpu_gate():
    evidence = run_gate("cpu")
    assert len(evidence["cases"]) == 18
    assert all(case["max_error"] == 0.0 for case in evidence["cases"])
    assert evidence["simple_point_maze_goal_distinction"]["cached_goal_image_byte_exact"]
