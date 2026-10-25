# Object-permanence source inventory

Pinned source: `824baddcdad087810d35638bf8f273b884f74975`; 150 task groups, 197 member scripts.

Regenerate with `python -m scripts.export.object_permanence inventory --source-root <checkout> --output <file>`.
This is a static, script-wide inventory. Unused builders can contribute signals;
absence of a signal does not prove support. Counts include all members.
Only explicitly listed nominal and additional members have audited preparation entry points;
run export to check their evaluated geometry, transforms, materials and camera at every frame.
Diversity variants, sampled scene semantics and downstream accuracy are not certified here.

The measured nominal column comes from the separately recorded [source audit](object-permanence-audit.json), not from static hints. It covers first nominal members; additionally certified members are listed separately in that audit.

## Capability triage

Groups cover every member. Static groups are conservative AST hints, not proof of support.
Known source-fidelity failures form a separate measured deferral queue.
Unused helpers may trigger a flag. Remaining priority is audited nominal, visibility/appearance,
orthographic, geometry/scale, physics bake, then rigid candidate requiring entry audit.
An applied, time-invariant modifier is allowed; an animated/deforming mesh is not.

| Review group | Members |
|---|---:|
| appearance-or-visibility-review | 7 |
| audited-nominal-runtime-gate | 7 |
| geometry-or-scale-review | 1 |
| orthographic-camera-review | 30 |
| rigid-candidate-needs-entry-audit | 7 |
| source-fidelity-review | 4 |
| versioned-visibility-runtime-gate | 141 |

## Source tasks

| Task | Members | Script-wide signals | First-member review group | Nominal preparation coverage | Measured nominal audit |
|---|---:|---|---|---|---|
| G01 hole_box_drop | 2 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G02 ramp_tunnel | 3 | location, modifiers, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G03 ramp_ball_blocked_by_wall | 2 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G04 high_low_cover | 2 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G05 car_vs_barrier | 7 | hide_render, hide_viewport, location, rotation_euler, scale | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G06 car_on_bridge | 3 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G07 guillotine_gate_stops | 2 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G08 drawer_moves_hidden_object | 2 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G09 low_beam_car_height | 3 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G10 size_gate | 2 | location, modifiers, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G11 box_holds_N_objects | 3 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G12 partition_box | 3 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G13 three_drawers | 5 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G14 u_tube_three_lanes | 6 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G15 ramp_panel_occludes_objects | 2 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G16 window_slit_mask_reveals | 2 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G17 moving_tray | 2 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G18 turntable_behind_screen | 4 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G19 row_screen_hides_objects | 3 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G20 connected_vertical_panel | 2 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G21 connected_sliding_doors | 2 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G22 top_rail_screen | 2 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G23 cabinet | 2 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G24 pivoting_occluder | 3 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G25 no_top_side_post_panel | 2 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G26 bottom_rail_screen | 2 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G27 support_removed_then_fall | 1 | location, rotation_euler | audited-nominal-runtime-gate | nominal export available | certified |
| G28 open_ended_tunnel | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G29 container_entry_left_u | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G30 hidden_object_moves_with_cart | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G31 two_lane_tunnel | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G32 cart_swap | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G33 partial_window_parallel_cars | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G34 occ_small_ball_behind_screen | 1 | location, rotation_euler | audited-nominal-runtime-gate | nominal export available | certified |
| G35 single_low_window | 1 | hide_render, hide_viewport, location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G36 rollercoaster_u_track_glassbox | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G37 u_track_occluded_ball | 1 | location, rotation_euler | audited-nominal-runtime-gate | nominal export available | certified |
| G38 pendulum_occluded_by_screen | 1 | location, rotation_euler | audited-nominal-runtime-gate | nominal export available | certified |
| G39 lidded_box_ball_enters | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G40 guided_elevator_hidden_ball | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G41 trapdoor_opens_ball_falls | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G42 ring_track_behind_center_block | 1 | dimensions, location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G43 three_balls_parallel_tunnels | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G44 two_balls_collide_and_bounce | 1 | dimensions, location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G45 opaque_lid_front_panel_drops | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G46 marked_boxes_swap | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G47 theater_curtain | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G48 sliding_window_row | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G49 two_supported_screens_close | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G50 ramp_ball_slotted_screen | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G51 vertical_panel_two_objects | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G52 ramp_open_right_gated_box | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G53 no_gap_top_rail_curtain | 1 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G54 serpentine_ramp_tunnel | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G55 rising_floor_screen | 1 | location | audited-nominal-runtime-gate | nominal export available | certified |
| G56 three_cup_shell_game | 1 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G57 ramp_ball_deflected_angled_wall | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G58 two_ball_trapdoor_size_filter | 1 | location, modifiers, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G59 hinged_lid_box_opens | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G60 newtons_cradle | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G61 spiral_ramp_behind_column | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G62 two_balls_cross_tunnel | 1 | location, rotation_euler | audited-nominal-runtime-gate | nominal export available | certified |
| G63 comb_occluder_sweep | 1 | location | source-fidelity-review | source fidelity deferred | certified |
| G64 flipboard_occluder | 1 | location, modifiers, rotation_euler | audited-nominal-runtime-gate | nominal export available | certified |
| G65 box_two_balls_relocate | 1 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G66 rotating_carousel_cups | 1 | location, rotation_euler | source-fidelity-review | source fidelity deferred | certified |
| G67 double_deflector_zigzag | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G68 width_slot_wall_two_balls | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G69 two_tier_trapdoor_cascade | 1 | location, modifiers, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G70 popaway_support_columns | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G71 double_doors_swing_open | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G72 sliding_lid_box | 1 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G73 break_scatter_cluster | 1 | location, rigid_body.kinematic, rigidbody, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G74 glancing_oblique_collision | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G75 ball_behind_rotating_billboard | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G76 pendulum_behind_post | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G77 three_balls_one_wide_tunnel | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G78 rolling_disc_tunnel | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G79 venetian_blinds | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G80 sliding_double_doors | 1 | location | source-fidelity-review | source fidelity deferred | certified |
| G81 rolling_shutter | 1 | location | source-fidelity-review | source fidelity deferred | certified |
| G82 accordion_fold_screen | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G83 four_cup_three_swaps | 1 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G84 two_balls_three_cups | 1 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G85 nested_cup_transfer | 1 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G86 conveyor_covered_boxes | 1 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G87 four_cup_carousel | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G88 shell_game_fakeout_reveal | 1 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G89 limbo_height_bar | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G90 funnel_size_sorter | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G91 turnstile_timed_gate | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G92 bumper_carom | 1 | location, rotation_axis_angle | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G93 tipping_shelf_drop | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G94 trapdoor_drop | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G95 conveyor_edge_fall | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G96 clamshell_box | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G97 rolltop_tambour | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G98 liftoff_dome_lid | 1 | location, modifiers, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G99 headon_velocity_exchange | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G100 heavy_light_collision | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G101 picket_fence_flicker | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G102 drop_screen_occluder | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G103 corner_turn_occlusion | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G104 bifold_concertina_doors | 1 | rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G105 descending_dome_cover | 1 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G106 rotating_drum_occluder | 1 | rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G107 sliding_cup_relocate | 1 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G108 two_carts_cross_swap | 1 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G109 tilting_tray_relocate | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G110 one_way_flap_gate | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G111 portcullis_drop_gate | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G112 banked_quarter_pipe_redirect | 1 | location, rotation_axis_angle | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G113 whipped_away_card | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G114 retractable_support_pins | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G115 steep_slope_slide | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G116 vault_swing_door | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G117 blooming_petal_box | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G118 matchbox_drawer | 1 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G119 offcenter_break_vsplit | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G120 pendulum_strike_projectile | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G121 wiper_screen_occlusion | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G122 ball_behind_box_stack | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G123 sliding_cover_panel | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G124 rising_sleeve_cover | 1 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G125 two_carts_reveal_empty | 1 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G126 turntable_two_boxes | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G127 rising_bollard_stop | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G128 swing_arm_barrier_stop | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G129 sliding_hatch_drop | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G130 bomb_bay_doors_drop | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G131 stacked_drawers_reveal | 1 | location | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G132 twist_open_capsule | 1 | location, modifiers, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G133 ball_topples_block | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G134 knock_ball_off_tee | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G135 lever_launch_transfer | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G136 ball_strikes_pendulum | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G137 pool_rack_break | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G138 bank_shot_carom | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G139 ball_shoves_block_slide | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G140 light_ball_rebounds_off_heavy | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G141 glancing_billiard_split | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G142 knock_ball_off_ledge | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G143 topple_two_blocks_apart | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G144 double_flap_top_box | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G145 tilt_platform_rolloff | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G146 hanging_ball_release_drop | 1 | location, rotation_euler, scale | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G147 drop_leaf_shelf | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G148 latch_release_flap_drop | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G149 snap_pillar_topple | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
| G150 rollers_part_drop | 1 | location, rotation_euler | versioned-visibility-runtime-gate | versioned ray contract only | certified |
