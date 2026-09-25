# Ratios derived from the preceding measurements

There are 60 available renderer/device rows covering 37 registrations: 25 have matched existing component measurements, while 35 combine Apple M5 CPU operations with Oscar GPU rendering and are explicitly arithmetic-only. The new composed run measures matched components directly. Literal requested sums are retained in JSON; the table avoids counting projection twice because the earlier public render(full State) already includes it.

| Registration / device | Matched components? | Get/step | Set/step | Get/rendered | Set/rendered | Get slowdown | Set slowdown |
|---|---|---:|---:|---:|---:|---:|---:|
| swm/AcrobotControl-v1 / cpu | yes | 0.560 | 0.558 | 0.018 | 0.018 | 1.018 | 1.018 |
| swm/AcrobotControl-v1 / gpu | NO: cross-host arithmetic | 0.560 | 0.558 | 0.011 | 0.011 | 1.011 | 1.011 |
| swm/AcrobotDMControl-v0 / cpu | yes | 0.544 | 0.612 | 0.025 | 0.028 | 1.025 | 1.028 |
| swm/AcrobotDMControl-v0 / gpu | NO: cross-host arithmetic | 0.544 | 0.612 | 0.022 | 0.025 | 1.022 | 1.025 |
| swm/BallInCupDMControl-v0 / cpu | yes | 0.449 | 0.526 | 0.004 | 0.004 | 1.004 | 1.004 |
| swm/BallInCupDMControl-v0 / gpu | NO: cross-host arithmetic | 0.449 | 0.526 | 0.022 | 0.026 | 1.022 | 1.026 |
| swm/CartPoleControl-v1 / cpu | yes | 0.703 | 0.630 | 0.024 | 0.022 | 1.024 | 1.022 |
| swm/CartPoleControl-v1 / gpu | NO: cross-host arithmetic | 0.703 | 0.630 | 0.015 | 0.013 | 1.015 | 1.013 |
| swm/CartpoleDMControl-v0 / cpu | yes | 0.491 | 0.487 | 0.037 | 0.037 | 1.037 | 1.037 |
| swm/CartpoleDMControl-v0 / gpu | NO: cross-host arithmetic | 0.491 | 0.487 | 0.022 | 0.022 | 1.022 | 1.022 |
| swm/CheetahDMControl-v0 / cpu | yes | 0.514 | 0.601 | 0.016 | 0.018 | 1.016 | 1.018 |
| swm/CheetahDMControl-v0 / gpu | NO: cross-host arithmetic | 0.514 | 0.601 | 0.078 | 0.091 | 1.078 | 1.091 |
| swm/CraftaxClassicPixels-v1 / cpu | yes | 0.083 | 1.077 | 0.035 | 0.458 | 1.035 | 1.458 |
| swm/CraftaxClassicPixels-v1 / gpu | yes | 0.235 | 0.998 | 0.097 | 0.411 | 1.097 | 1.411 |
| swm/CraftaxPixels-v1 / cpu | yes | 0.096 | 1.016 | 0.039 | 0.407 | 1.039 | 1.407 |
| swm/CraftaxPixels-v1 / gpu | yes | 0.298 | 1.062 | 0.111 | 0.397 | 1.111 | 1.397 |
| swm/FetchPickAndPlace-v3 / gpu | NO: cross-host arithmetic | 0.183 | 0.259 | 0.033 | 0.046 | 1.033 | 1.046 |
| swm/FetchPickAndPlaceDense-v3 / gpu | NO: cross-host arithmetic | 0.187 | 0.259 | 0.033 | 0.046 | 1.033 | 1.046 |
| swm/FetchPickAndPlaceDict-v3 / gpu | NO: cross-host arithmetic | 0.183 | 0.264 | 0.032 | 0.046 | 1.032 | 1.046 |
| swm/FetchPush-v3 / gpu | NO: cross-host arithmetic | 0.171 | 0.242 | 0.032 | 0.046 | 1.032 | 1.046 |
| swm/FetchPushDense-v3 / gpu | NO: cross-host arithmetic | 0.173 | 0.247 | 0.032 | 0.046 | 1.032 | 1.046 |
| swm/FetchPushDict-v3 / gpu | NO: cross-host arithmetic | 0.177 | 0.269 | 0.034 | 0.051 | 1.034 | 1.051 |
| swm/FetchReach-v3 / gpu | NO: cross-host arithmetic | 0.202 | 0.280 | 0.034 | 0.047 | 1.034 | 1.047 |
| swm/FetchReachDense-v3 / gpu | NO: cross-host arithmetic | 0.200 | 0.301 | 0.034 | 0.051 | 1.034 | 1.051 |
| swm/FetchReachDict-v3 / gpu | NO: cross-host arithmetic | 0.204 | 0.290 | 0.034 | 0.049 | 1.034 | 1.049 |
| swm/FetchSlide-v3 / gpu | NO: cross-host arithmetic | 0.151 | 0.231 | 0.032 | 0.048 | 1.032 | 1.048 |
| swm/FetchSlideDense-v3 / gpu | NO: cross-host arithmetic | 0.159 | 0.234 | 0.033 | 0.048 | 1.033 | 1.048 |
| swm/FetchSlideDict-v3 / gpu | NO: cross-host arithmetic | 0.152 | 0.237 | 0.032 | 0.049 | 1.032 | 1.049 |
| swm/FingerDMControl-v0 / cpu | yes | 0.466 | 0.526 | 0.013 | 0.014 | 1.013 | 1.014 |
| swm/FingerDMControl-v0 / gpu | NO: cross-host arithmetic | 0.466 | 0.526 | 0.027 | 0.031 | 1.027 | 1.031 |
| swm/HopperDMControl-v0 / cpu | yes | 0.473 | 0.538 | 0.022 | 0.025 | 1.022 | 1.025 |
| swm/HopperDMControl-v0 / gpu | NO: cross-host arithmetic | 0.473 | 0.538 | 0.079 | 0.090 | 1.079 | 1.090 |
| swm/HumanoidDMControl-v0 / cpu | yes | 0.260 | 0.328 | 0.008 | 0.010 | 1.008 | 1.010 |
| swm/HumanoidDMControl-v0 / gpu | NO: cross-host arithmetic | 0.260 | 0.328 | 0.064 | 0.081 | 1.064 | 1.081 |
| swm/ManipulatorDMControl-v0 / cpu | yes | 0.331 | 0.385 | 0.009 | 0.011 | 1.009 | 1.011 |
| swm/ManipulatorDMControl-v0 / gpu | NO: cross-host arithmetic | 0.331 | 0.385 | 0.028 | 0.032 | 1.028 | 1.032 |
| swm/MountainCarContinuousControl-v0 / cpu | yes | 0.714 | 0.599 | 0.020 | 0.016 | 1.020 | 1.016 |
| swm/MountainCarContinuousControl-v0 / gpu | NO: cross-host arithmetic | 0.714 | 0.599 | 0.011 | 0.010 | 1.011 | 1.010 |
| swm/MountainCarControl-v0 / cpu | yes | 0.632 | 0.592 | 0.016 | 0.015 | 1.016 | 1.015 |
| swm/MountainCarControl-v0 / gpu | NO: cross-host arithmetic | 0.632 | 0.592 | 0.009 | 0.009 | 1.009 | 1.009 |
| swm/OGBCube-v0 / gpu | NO: cross-host arithmetic | 0.608 | 141.919 | 0.031 | 7.307 | 1.031 | 8.307 |
| swm/OGBMaze-v0 / cpu | yes | 0.382 | 0.920 | 0.138 | 0.332 | 1.138 | 1.332 |
| swm/OGBMaze-v0 / gpu | NO: cross-host arithmetic | 0.382 | 0.920 | 0.033 | 0.080 | 1.033 | 1.080 |
| swm/OGBScene-v0 / gpu | NO: cross-host arithmetic | 0.510 | 146.427 | 0.029 | 8.278 | 1.029 | 9.278 |
| swm/PendulumControl-v1 / cpu | yes | 0.667 | 0.594 | 0.032 | 0.028 | 1.032 | 1.028 |
| swm/PendulumControl-v1 / gpu | NO: cross-host arithmetic | 0.667 | 0.594 | 0.012 | 0.010 | 1.012 | 1.010 |
| swm/PendulumDMControl-v0 / cpu | yes | 0.603 | 0.677 | 0.024 | 0.027 | 1.024 | 1.027 |
| swm/PendulumDMControl-v0 / gpu | NO: cross-host arithmetic | 0.603 | 0.677 | 0.020 | 0.022 | 1.020 | 1.022 |
| swm/Piecewise-v0 / cpu | yes | 0.359 | 0.317 | 0.153 | 0.135 | 1.153 | 1.135 |
| swm/Piecewise-v0 / gpu | NO: cross-host arithmetic | 0.359 | 0.317 | 0.048 | 0.043 | 1.048 | 1.043 |
| swm/QuadrupedDMControl-v0 / cpu | yes | 0.316 | 0.390 | 0.004 | 0.005 | 1.004 | 1.005 |
| swm/QuadrupedDMControl-v0 / gpu | NO: cross-host arithmetic | 0.316 | 0.390 | 0.036 | 0.044 | 1.036 | 1.044 |
| swm/ReacherDMControl-v0 / cpu | yes | 0.587 | 0.672 | 0.024 | 0.027 | 1.024 | 1.027 |
| swm/ReacherDMControl-v0 / gpu | NO: cross-host arithmetic | 0.587 | 0.672 | 0.020 | 0.023 | 1.020 | 1.023 |
| swm/SimplePointMaze-v0 / cpu | yes | 0.425 | 32.680 | 0.044 | 3.350 | 1.044 | 4.350 |
| swm/SimplePointMaze-v0 / gpu | NO: cross-host arithmetic | 0.425 | 32.680 | 0.030 | 2.296 | 1.030 | 3.296 |
| swm/TwoRoom-v1 / cpu | yes | 0.210 | 0.189 | 0.117 | 0.105 | 1.117 | 1.105 |
| swm/TwoRoom-v1 / gpu | NO: cross-host arithmetic | 0.210 | 0.189 | 0.030 | 0.027 | 1.030 | 1.027 |
| swm/WalkerDMControl-v0 / cpu | yes | 0.353 | 0.414 | 0.015 | 0.018 | 1.015 | 1.018 |
| swm/WalkerDMControl-v0 / gpu | NO: cross-host arithmetic | 0.353 | 0.414 | 0.072 | 0.085 | 1.072 | 1.085 |
