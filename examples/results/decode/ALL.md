# One frame, three ways — 196 environments

Every column is frames/s for the same transformation, `state -> uint8 frame`, at the env's own
native resolution.

| column | what it is |
|---|---|
| **default** | the env's own official renderer, one frame at a time — the only entry point it offers. Blank where the reference package is not installed in the interpreter that owns that env. |
| **ours** | the GPU path at its best batch. |
| **stored** | one CPU core decoding *this env's own frame*, JPEG q90, already in memory — no filesystem, no network. The case most favourable to a stored dataset. |

`cores` is **ours / stored**: how many CPU cores of decoding one GPU replaces.

`ss` is the supersample the env resolved to (its own `default_supersample`, not a flag), and
`cpu%` is process CPU seconds per wall second of the timed loop: at or above 100% the host is
busy for the whole render and that row is a measurement of the node's processor under its load,
not of the GPU. The ray-cast families sit there.

| env | family | frame | ss | default | ours (batch) | cpu% | stored /core | **cores** | vs default |
|---|---|---|---|---|---|---|---|---|---|
| GriddlyCookMePasta | Griddly | 264x336 | 1 | 147 | 5,898,273 (16,384) | 101% | 1,524 | **3,869.5x** | 40,035x |
| GriddlySokoban | Griddly | 216x312 | 1 | 144 | 6,681,328 (16,384) | 101% | 2,111 | **3,165.4x** | 46,430x |
| GriddlyClusters | Griddly | 240x312 | 1 | 148 | 6,056,032 (16,384) | 100% | 1,980 | **3,058.9x** | 40,923x |
| OvercookedCounterCircuit | Overcooked | 224x320 | 1 | 860 | 6,974,633 (16,384) | 100% | 2,511 | **2,777.8x** | 8,107x |
| GriddlyButterflies | Griddly | 264x672 | 1 | 102 | 2,547,779 (4,096) | 100% | 953 | **2,673.9x** | 25,096x |
| OvercookedCoordRing | Overcooked | 224x224 | 1 | 1,189 | 6,968,629 (16,384) | 100% | 2,709 | **2,572.8x** | 5,860x |
| GriddlyLabyrinth | Griddly | 336x384 | 1 | 121 | 2,311,257 (16,384) | 102% | 910 | **2,539.5x** | 19,157x |
| Sokoban | Sokoban | 160x160 | 1 | 256 | 9,572,010 (16,384) | 100% | 4,186 | **2,286.6x** | 37,396x |
| GriddlySokoban2 | Griddly | 168x192 | 1 | 124 | 8,316,502 (16,384) | 101% | 3,775 | **2,203.3x** | 67,030x |
| OvercookedCrampedRoom | Overcooked | 192x224 | 1 | 472 | 8,055,812 (16,384) | 100% | 3,715 | **2,168.4x** | 17,065x |
| GriddlyBait | Griddly | 144x120 | 1 | 171 | 9,473,899 (16,384) | 100% | 4,790 | **1,977.9x** | 55,394x |
| OvercookedForcedCoord | Overcooked | 224x224 | 1 | 1,148 | 6,967,233 (16,384) | 100% | 3,974 | **1,753.4x** | 6,067x |
| OvercookedAsymmAdvantages | Overcooked | 224x352 | 1 | 27 | 3,596,911 (16,384) | 100% | 3,401 | **1,057.7x** | 132,228x |
| MiniGridFourRooms | MiniGrid / BabyAI | 608x608 | 1 | 238 | 711,102 (4,096) | 100% | 800 | **889.1x** | 2,985x |
| MiniGridPOVFetch | MiniGrid / BabyAI | 224x224 | 1 | 447 | 4,420,748 (4,096) | 100% | 5,096 | **867.4x** | 9,896x |
| MiniGridDoorKey | MiniGrid / BabyAI | 256x256 | 1 | 392 | 1,211,344 (4,096) | 100% | 2,237 | **541.4x** | 3,093x |
| MiniGridKeyCorridor | MiniGrid / BabyAI | 320x320 | 1 | 206 | 1,153,969 (4,096) | 100% | 2,654 | **434.8x** | 5,610x |
| MiniGridLavaCrossing | MiniGrid / BabyAI | 288x288 | 1 | 938 | 1,179,476 (4,096) | 100% | 2,906 | **405.9x** | 1,257x |
| JumanjiMinesweeper | Jumanji | 200x200 | 1 | 11 | 1,196,463 (16,384) | 100% | 3,405 | **351.3x** | 108,235x |
| JumanjiSudoku | Jumanji | 200x200 | 1 | 12 | 1,282,419 (16,384) | 100% | 3,708 | **345.8x** | 110,727x |
| MiniHack | MiniHack | 336x1264 | 1 | — | 244,515 (4,096) | 100% | 721 | **339.0x** | — |
| BabyAIGoToLocal | MiniGrid / BabyAI | 256x256 | 1 | 395 | 1,235,817 (4,096) | 100% | 3,730 | **331.3x** | 3,129x |
| MiniGridEmpty | MiniGrid / BabyAI | 256x256 | 1 | 854 | 1,303,556 (4,096) | 100% | 4,027 | **323.7x** | 1,527x |
| JumanjiSudokuVeryEasy | Jumanji | 200x200 | 1 | 10 | 1,282,433 (16,384) | 100% | 3,984 | **321.9x** | 127,304x |
| ProcgenMaze | Procgen | 64x64 | 1 | 896 | 1,576,433 (4,096) | 100% | 8,138 | **193.7x** | 1,759x |
| JumanjiRubiksCubePartlyScrambled | Jumanji | 200x200 | 1 | 8 | 593,505 (16,384) | 100% | 3,562 | **166.6x** | 77,790x |
| JumanjiRubiksCube | Jumanji | 200x200 | 1 | 8 | 593,746 (16,384) | 100% | 3,612 | **164.4x** | 77,785x |
| PushT | PushT | 96x96 | 1 | 476 | 1,181,957 (4,096) | 100% | 7,699 | **153.5x** | 2,481x |
| JumanjiMaze | Jumanji | 200x200 | 1 | 82 | 564,824 (16,384) | 100% | 3,892 | **145.1x** | 6,873x |
| JumanjiSokoban | Jumanji | 200x200 | 1 | 6 | 403,573 (16,384) | 100% | 2,829 | **142.7x** | 63,367x |
| ProcgenMiner | Procgen | 64x64 | 1 | 974 | 1,156,551 (4,096) | 100% | 9,483 | **122.0x** | 1,188x |
| JumanjiConnector | Jumanji | 200x200 | 1 | 4 | 455,587 (16,384) | 100% | 3,768 | **120.9x** | 120,792x |
| CraftaxClassic | Craftax | 112x144 | 1 | — | 568,670 (4,096) | 100% | 5,544 | **102.6x** | — |
| PointmazeTeleportStitch | OGBench | 224x224 | 2 | 454 | 241,443 (4,096) | 100% | 2,504 | **96.4x** | 532x |
| PointmazeTeleportNavigate | OGBench | 224x224 | 2 | 439 | 241,667 (4,096) | 100% | 3,078 | **78.5x** | 550x |
| PointmazeGiantNavigate | OGBench | 224x224 | 2 | 406 | 242,264 (4,096) | 100% | 3,120 | **77.6x** | 596x |
| Piecewise | swm nav | 224x224 | 1 | 830 | 187,975 (4,096) | 100% | 2,494 | **75.4x** | 226x |
| Merge | highway-env | 150x600 | 1 | — | 198,437 (4,096) | 100% | 2,862 | **69.3x** | — |
| PointmazeGiantStitch | OGBench | 224x224 | 2 | 414 | 242,479 (4,096) | 100% | 3,525 | **68.8x** | 586x |
| PointmazeMediumStitch | OGBench | 224x224 | 2 | 488 | 242,276 (4,096) | 100% | 3,522 | **68.8x** | 497x |
| Crafter | Crafter | 64x64 | 1 | — | 661,255 (4,096) | 100% | 9,648 | **68.5x** | — |
| Highway | highway-env | 150x600 | 1 | — | 199,371 (4,096) | 100% | 2,926 | **68.1x** | — |
| PointmazeLargeNavigate | OGBench | 224x224 | 2 | 436 | 242,332 (4,096) | 100% | 4,080 | **59.4x** | 555x |
| PointmazeLargeStitch | OGBench | 224x224 | 2 | 450 | 242,672 (4,096) | 100% | 4,086 | **59.4x** | 539x |
| PointmazeMediumNavigate | OGBench | 224x224 | 2 | 469 | 241,664 (4,096) | 100% | 4,218 | **57.3x** | 515x |
| SimplePointMaze | swm nav | 500x500 | 1 | 53 | 70,746 (4,096) | 100% | 1,317 | **53.7x** | 1,323x |
| ProcgenChaser | Procgen | 64x64 | 1 | 873 | 347,350 (4,096) | 100% | 7,511 | **46.2x** | 398x |
| PgxTicTacToe | PGX | 240x240 | 1 | 97 | 107,858 (4,096) | 100% | 2,493 | **43.3x** | 1,109x |
| Parking | highway-env | 300x600 | 1 | — | 62,138 (4,096) | 100% | 1,507 | **41.2x** | — |
| PointMazeOpen | D4RL PointMaze | 224x224 | 2 | 2,131 | 94,813 (4,096) | 100% | 2,355 | **40.3x** | 44x |
| PgxGo9x9 | PGX | 250x250 | 1 | 40 | 92,855 (4,096) | 100% | 2,336 | **39.8x** | 2,319x |
| AntmazeTeleportExplore | OGBench | 224x224 | 2 | 411 | 90,246 (4,096) | 287% | 2,368 | **38.1x** | 220x |
| AntmazeLargeNavigate | OGBench | 224x224 | 2 | 442 | 90,635 (4,096) | 287% | 2,422 | **37.4x** | 205x |
| AntmazeMediumExplore | OGBench | 224x224 | 2 | 448 | 92,149 (4,096) | 290% | 2,615 | **35.2x** | 206x |
| PiecewiseZones | swm nav | 224x224 | 1 | 501 | 141,119 (4,096) | 100% | 4,075 | **34.6x** | 281x |
| Intersection | highway-env | 600x600 | 1 | — | 25,311 (1,024) | 100% | 794 | **31.9x** | — |
| PointMazeMedium | D4RL PointMaze | 224x224 | 2 | 1,988 | 94,870 (4,096) | 100% | 3,104 | **30.6x** | 48x |
| AntmazeLargeStitch | OGBench | 224x224 | 2 | 441 | 90,808 (4,096) | 287% | 3,102 | **29.3x** | 206x |
| ProcgenHeist | Procgen | 64x64 | 1 | 952 | 316,300 (4,096) | 100% | 10,815 | **29.2x** | 332x |
| AntmazeTeleportNavigate | OGBench | 224x224 | 2 | 425 | 90,294 (4,096) | 288% | 3,090 | **29.2x** | 212x |
| Pendulum | classic control | 500x500 | 1 | 283 | 38,396 (4,096) | 100% | 1,318 | **29.1x** | 136x |
| AntmazeLargeExplore | OGBench | 224x224 | 2 | 429 | 90,728 (4,096) | 288% | 3,129 | **29.0x** | 212x |
| Acrobot | classic control | 500x500 | 1 | 408 | 31,630 (4,096) | 100% | 1,101 | **28.7x** | 77x |
| Racetrack | highway-env | 600x600 | 1 | — | 24,573 (1,024) | 100% | 860 | **28.6x** | — |
| HumanoidmazeLargeNavigate | OGBench | 224x224 | 2 | 438 | 79,578 (4,096) | 267% | 2,864 | **27.8x** | 182x |
| MountainCarContinuous | classic control | 400x600 | 1 | 452 | 31,137 (4,096) | 100% | 1,148 | **27.1x** | 69x |
| PgxOthello | PGX | 270x270 | 1 | 21 | 81,162 (4,096) | 100% | 3,130 | **25.9x** | 3,935x |
| AntmazeTeleportStitch | OGBench | 224x224 | 2 | 425 | 89,997 (4,096) | 286% | 3,487 | **25.8x** | 212x |
| CartPole | classic control | 400x600 | 1 | 447 | 35,973 (4,096) | 100% | 1,441 | **25.0x** | 80x |
| PgxConnectFour | PGX | 280x280 | 1 | 67 | 77,336 (4,096) | 100% | 3,101 | **24.9x** | 1,158x |
| AntmazeGiantStitch | OGBench | 224x224 | 2 | 399 | 87,724 (4,096) | 281% | 3,548 | **24.7x** | 220x |
| PointMazeLarge | D4RL PointMaze | 224x224 | 2 | 1,803 | 95,050 (4,096) | 100% | 3,857 | **24.6x** | 53x |
| TwoRoom | swm nav | 224x224 | 1 | 668 | 117,007 (4,096) | 100% | 4,967 | **23.6x** | 175x |
| MountainCar | classic control | 400x600 | 1 | 415 | 31,111 (4,096) | 100% | 1,364 | **22.8x** | 75x |
| HumanoidmazeMediumStitch | OGBench | 224x224 | 2 | 461 | 81,013 (4,096) | 271% | 3,666 | **22.1x** | 176x |
| AntmazeMediumNavigate | OGBench | 224x224 | 2 | 465 | 92,167 (4,096) | 290% | 4,194 | **22.0x** | 198x |
| AntmazeMediumStitch | OGBench | 224x224 | 2 | 465 | 92,105 (4,096) | 291% | 4,238 | **21.7x** | 198x |
| AntmazeGiantNavigate | OGBench | 224x224 | 2 | 391 | 87,636 (4,096) | 280% | 4,040 | **21.7x** | 224x |
| PointMazeUMaze | D4RL PointMaze | 224x224 | 2 | 2,255 | 94,990 (4,096) | 100% | 4,379 | **21.7x** | 42x |
| Roundabout | highway-env | 600x600 | 1 | — | 16,538 (1,024) | 100% | 781 | **21.2x** | — |
| HumanoidmazeLargeStitch | OGBench | 224x224 | 2 | 450 | 79,658 (4,096) | 266% | 4,088 | **19.5x** | 177x |
| HumanoidmazeGiantStitch | OGBench | 224x224 | 2 | 399 | 77,507 (4,096) | 263% | 4,060 | **19.1x** | 194x |
| HumanoidmazeMediumNavigate | OGBench | 224x224 | 2 | 443 | 80,933 (4,096) | 268% | 4,251 | **19.0x** | 183x |
| HumanoidmazeGiantNavigate | OGBench | 224x224 | 2 | 391 | 77,494 (4,096) | 263% | 4,154 | **18.7x** | 198x |
| DMCCheetahRun | DM Control | 84x84 | 1 | 354 | 124,345 (4,096) | 100% | 7,357 | **16.9x** | 351x |
| TwoRoomTarget | swm nav | 224x224 | 1 | 477 | 76,785 (4,096) | 100% | 4,922 | **15.6x** | 161x |
| DMCHopperHop | DM Control | 84x84 | 1 | 336 | 124,436 (4,096) | 100% | 9,214 | **13.5x** | 371x |
| LunarLander | LunarLander | 400x600 | 1 | 339 | 18,825 (4,096) | 100% | 1,399 | **13.5x** | 55x |
| DMCWalkerWalk | DM Control | 84x84 | 1 | 336 | 116,662 (4,096) | 100% | 9,451 | **12.3x** | 347x |
| MujocoReacher | MuJoCo locomotion | 240x240 | 1 | 2,037 | 52,901 (4,096) | 100% | 4,351 | **12.2x** | 26x |
| MujocoInvertedPendulum | MuJoCo locomotion | 240x240 | 1 | 2,502 | 52,503 (4,096) | 100% | 4,645 | **11.3x** | 21x |
| DMCHumanoidStand | DM Control | 84x84 | 1 | 282 | 101,359 (4,096) | 100% | 9,556 | **10.6x** | 359x |
| FetchSlide | Fetch | 480x480 | 1 | 594 | 8,958 (4,096) | 100% | 850 | **10.5x** | 15x |
| FetchPickAndPlace | Fetch | 480x480 | 1 | 615 | 8,653 (4,096) | 100% | 857 | **10.1x** | 14x |
| FetchPickAndPlaceDict | Fetch | 480x480 | 1 | 706 | 8,655 (4,096) | 100% | 860 | **10.1x** | 12x |
| FetchReachDict | Fetch | 480x480 | 1 | 713 | 8,682 (4,096) | 100% | 874 | **9.9x** | 12x |
| MujocoInvertedDoublePendulum | MuJoCo locomotion | 240x240 | 1 | 2,115 | 43,658 (4,096) | 100% | 4,757 | **9.2x** | 21x |
| DMCCartpoleBalance | DM Control | 84x84 | 1 | 358 | 76,456 (4,096) | 252% | 8,691 | **8.8x** | 214x |
| DMCSwimmerSwimmer6 | DM Control | 84x84 | 1 | 272 | 67,351 (4,096) | 100% | 7,734 | **8.7x** | 247x |
| FetchPickAndPlaceDense | Fetch | 480x480 | 1 | 702 | 8,654 (4,096) | 100% | 1,000 | **8.7x** | 12x |
| DMCFingerSpin | DM Control | 84x84 | 1 | 364 | 61,848 (4,096) | 356% | 7,194 | **8.6x** | 170x |
| FetchReach | Fetch | 480x480 | 1 | 621 | 8,686 (4,096) | 100% | 1,026 | **8.5x** | 14x |
| FetchReachDense | Fetch | 480x480 | 1 | 703 | 8,678 (4,096) | 100% | 1,045 | **8.3x** | 12x |
| HumanoidBenchH1Slide | HumanoidBench | 256x256 | 1 | 60 | 17,112 (4,096) | 134% | 2,085 | **8.2x** | 288x |
| MujocoPusher | MuJoCo locomotion | 240x240 | 1 | 461 | 36,628 (4,096) | 1124% | 4,495 | **8.1x** | 79x |
| HumanoidBenchH1HandSlide | HumanoidBench | 256x256 | 1 | 57 | 15,481 (4,096) | 131% | 1,916 | **8.1x** | 270x |
| HumanoidBenchH1HandRun | HumanoidBench | 256x256 | 1 | 38 | 17,481 (4,096) | 135% | 2,188 | **8.0x** | 462x |
| DMCCartpoleSwingup | DM Control | 84x84 | 1 | 351 | 74,802 (4,096) | 249% | 9,437 | **7.9x** | 213x |
| FetchSlideDense | Fetch | 480x480 | 1 | 591 | 8,953 (4,096) | 100% | 1,140 | **7.9x** | 15x |
| FetchPushDict | Fetch | 480x480 | 1 | 678 | 8,749 (4,096) | 100% | 1,115 | **7.8x** | 13x |
| DMCPointMassEasy | DM Control | 84x84 | 1 | 366 | 59,302 (4,096) | 100% | 7,763 | **7.6x** | 162x |
| FetchSlideDict | Fetch | 480x480 | 1 | 586 | 8,956 (4,096) | 100% | 1,228 | **7.3x** | 15x |
| FetchPushDense | Fetch | 480x480 | 1 | 672 | 8,746 (4,096) | 100% | 1,250 | **7.0x** | 13x |
| FetchPush | Fetch | 480x480 | 1 | 600 | 8,744 (4,096) | 100% | 1,255 | **7.0x** | 15x |
| DMCPendulumSwingup | DM Control | 84x84 | 1 | 356 | 67,711 (4,096) | 370% | 10,514 | **6.4x** | 190x |
| HumanoidBenchH1HandWalk | HumanoidBench | 256x256 | 1 | 38 | 17,471 (4,096) | 135% | 2,891 | **6.0x** | 462x |
| HumanoidBenchH1HandStand | HumanoidBench | 256x256 | 1 | 38 | 17,459 (4,096) | 135% | 2,915 | **6.0x** | 461x |
| HumanoidBenchH1Stair | HumanoidBench | 256x256 | 1 | 59 | 13,214 (4,096) | 126% | 2,329 | **5.7x** | 225x |
| DMCFingerTurnHard | DM Control | 84x84 | 1 | 352 | 40,348 (4,096) | 268% | 7,466 | **5.4x** | 115x |
| DMCFingerTurnEasy | DM Control | 84x84 | 1 | 351 | 40,033 (4,096) | 267% | 7,537 | **5.3x** | 114x |
| HumanoidBenchH1Walk | HumanoidBench | 256x256 | 1 | 39 | 19,441 (4,096) | 139% | 3,856 | **5.0x** | 503x |
| HumanoidBenchH1Stand | HumanoidBench | 256x256 | 1 | 39 | 19,462 (4,096) | 139% | 3,903 | **5.0x** | 503x |
| HumanoidBenchH1HandSitHard | HumanoidBench | 256x256 | 1 | 38 | 12,023 (4,096) | 124% | 2,411 | **5.0x** | 319x |
| HumanoidBenchH1Run | HumanoidBench | 256x256 | 1 | 39 | 19,496 (4,096) | 139% | 3,920 | **5.0x** | 505x |
| RoboCasaCounterToSinkWrist | RoboCasa | 256x256 | 1 | 546 | 13,326 (1,024) | 100% | 2,710 | **4.9x** | 24x |
| DMCAcrobotSwingup | DM Control | 84x84 | 1 | 361 | 48,877 (4,096) | 292% | 10,110 | **4.8x** | 136x |
| MetaWorldAssembly | Meta-World | 480x480 | 1 | 214 | 2,879 (4,096) | 100% | 601 | **4.8x** | 13x |
| DMCReacherHard | DM Control | 84x84 | 1 | 355 | 48,812 (4,096) | 295% | 10,433 | **4.7x** | 138x |
| MetaWorldPushBack | Meta-World | 480x480 | 1 | 219 | 2,961 (4,096) | 100% | 633 | **4.7x** | 14x |
| DMCReacherEasy | DM Control | 84x84 | 1 | 358 | 47,961 (4,096) | 291% | 10,380 | **4.6x** | 134x |
| MetaWorldBasketball | Meta-World | 480x480 | 1 | 215 | 2,808 (4,096) | 100% | 643 | **4.4x** | 13x |
| MujocoHalfCheetah | MuJoCo locomotion | 240x240 | 1 | 369 | 10,445 (4,096) | 121% | 2,430 | **4.3x** | 28x |
| MetaWorldPush | Meta-World | 480x480 | 1 | 217 | 2,960 (4,096) | 100% | 689 | **4.3x** | 14x |
| MetaWorldPegUnplugSide | Meta-World | 480x480 | 1 | 212 | 2,829 (4,096) | 100% | 663 | **4.3x** | 13x |
| MetaWorldReachWall | Meta-World | 480x480 | 1 | 218 | 2,940 (4,096) | 100% | 690 | **4.3x** | 14x |
| HumanoidBenchH1BalanceSimple | HumanoidBench | 256x256 | 1 | 39 | 12,588 (4,096) | 125% | 2,960 | **4.3x** | 326x |
| MetaWorldFaucetClose | Meta-World | 480x480 | 1 | 213 | 2,839 (4,096) | 100% | 690 | **4.1x** | 13x |
| MetaWorldHandInsert | Meta-World | 480x480 | 1 | 173 | 2,954 (4,096) | 100% | 731 | **4.0x** | 17x |
| MetaWorldDoorUnlock | Meta-World | 480x480 | 1 | 215 | 2,616 (4,096) | 100% | 655 | **4.0x** | 12x |
| MetaWorldDrawerClose | Meta-World | 480x480 | 1 | 219 | 2,831 (4,096) | 100% | 713 | **4.0x** | 13x |
| MetaWorldShelfPlace | Meta-World | 480x480 | 1 | 218 | 2,739 (4,096) | 100% | 691 | **4.0x** | 13x |
| MetaWorldDrawerOpen | Meta-World | 480x480 | 1 | 221 | 2,845 (4,096) | 100% | 720 | **4.0x** | 13x |
| HumanoidBenchH1HandBalanceSimple | HumanoidBench | 256x256 | 1 | 38 | 11,280 (4,096) | 123% | 2,883 | **3.9x** | 299x |
| MetaWorldBoxClose | Meta-World | 480x480 | 1 | 216 | 2,858 (4,096) | 100% | 741 | **3.9x** | 13x |
| MetaWorldPushWall | Meta-World | 480x480 | 1 | 219 | 2,942 (4,096) | 100% | 767 | **3.8x** | 13x |
| MujocoAnt | MuJoCo locomotion | 240x240 | 1 | 352 | 9,727 (4,096) | 120% | 2,556 | **3.8x** | 28x |
| MetaWorldHandlePressSide | Meta-World | 480x480 | 1 | 216 | 2,707 (4,096) | 100% | 735 | **3.7x** | 13x |
| MetaWorldCoffeePush | Meta-World | 480x480 | 1 | 213 | 2,748 (4,096) | 100% | 753 | **3.7x** | 13x |
| MetaWorldDialTurn | Meta-World | 480x480 | 1 | 217 | 2,927 (4,096) | 100% | 803 | **3.6x** | 13x |
| MetaWorldCoffeePull | Meta-World | 480x480 | 1 | 214 | 2,754 (4,096) | 100% | 758 | **3.6x** | 13x |
| MetaWorldSweep | Meta-World | 480x480 | 1 | 220 | 2,966 (4,096) | 100% | 818 | **3.6x** | 13x |
| HumanoidBenchH1SitHard | HumanoidBench | 256x256 | 1 | 38 | 13,405 (4,096) | 127% | 3,704 | **3.6x** | 349x |
| HumanoidBenchH1SitSimple | HumanoidBench | 256x256 | 1 | 39 | 13,392 (4,096) | 127% | 3,729 | **3.6x** | 348x |
| MetaWorldPegInsertSide | Meta-World | 480x480 | 1 | 221 | 2,829 (4,096) | 100% | 795 | **3.6x** | 13x |
| MujocoSwimmer | MuJoCo locomotion | 240x240 | 1 | 352 | 10,395 (4,096) | 121% | 2,928 | **3.6x** | 29x |
| MetaWorldButtonPressWall | Meta-World | 480x480 | 1 | 214 | 2,744 (4,096) | 100% | 781 | **3.5x** | 13x |
| MetaWorldPickPlace | Meta-World | 480x480 | 1 | 217 | 2,959 (4,096) | 100% | 843 | **3.5x** | 14x |
| MujocoHumanoid | MuJoCo locomotion | 240x240 | 1 | 325 | 9,250 (4,096) | 119% | 2,655 | **3.5x** | 28x |
| MetaWorldCoffeeButton | Meta-World | 480x480 | 1 | 213 | 2,749 (4,096) | 100% | 790 | **3.5x** | 13x |
| MetaWorldPickOutOfHole | Meta-World | 480x480 | 1 | 213 | 2,848 (4,096) | 100% | 822 | **3.5x** | 13x |
| MetaWorldHammer | Meta-World | 480x480 | 1 | 214 | 2,779 (4,096) | 100% | 803 | **3.5x** | 13x |
| MujocoHopper | MuJoCo locomotion | 240x240 | 1 | 355 | 10,465 (4,096) | 121% | 3,057 | **3.4x** | 29x |
| CarRacing | CarRacing | 96x96 | 1 | 132 | 26,927 (4,096) | 100% | 8,118 | **3.3x** | 203x |
| MujocoWalker2d | MuJoCo locomotion | 240x240 | 1 | 370 | 10,499 (4,096) | 119% | 3,168 | **3.3x** | 28x |
| HumanoidBenchH1HandSitSimple | HumanoidBench | 256x256 | 1 | 38 | 12,118 (4,096) | 124% | 3,674 | **3.3x** | 321x |
| MetaWorldPlateSlideBackSide | Meta-World | 480x480 | 1 | 219 | 2,589 (4,096) | 100% | 810 | **3.2x** | 12x |
| MetaWorldButtonPressTopdown | Meta-World | 480x480 | 1 | 214 | 2,804 (4,096) | 100% | 882 | **3.2x** | 13x |
| MetaWorldPlateSlide | Meta-World | 480x480 | 1 | 215 | 2,569 (4,096) | 100% | 808 | **3.2x** | 12x |
| MetaWorldLeverPull | Meta-World | 480x480 | 1 | 221 | 2,770 (4,096) | 100% | 873 | **3.2x** | 13x |
| MetaWorldBinPicking | Meta-World | 480x480 | 1 | 213 | 2,893 (4,096) | 100% | 921 | **3.1x** | 14x |
| MetaWorldSweepInto | Meta-World | 480x480 | 1 | 220 | 2,977 (4,096) | 100% | 948 | **3.1x** | 14x |
| MetaWorldButtonPressTopdownWall | Meta-World | 480x480 | 1 | 215 | 2,792 (4,096) | 100% | 894 | **3.1x** | 13x |
| MetaWorldPickPlaceWall | Meta-World | 480x480 | 1 | 217 | 2,940 (4,096) | 100% | 945 | **3.1x** | 14x |
| MetaWorldFaucetOpen | Meta-World | 480x480 | 1 | 178 | 2,838 (4,096) | 100% | 912 | **3.1x** | 16x |
| MetaWorldReach | Meta-World | 480x480 | 1 | 216 | 2,959 (4,096) | 100% | 953 | **3.1x** | 14x |
| RoboCasaCounterToSinkRight | RoboCasa | 256x256 | 1 | 437 | 5,688 (4,096) | 100% | 1,833 | **3.1x** | 13x |
| MetaWorldSoccer | Meta-World | 480x480 | 1 | 220 | 2,407 (4,096) | 100% | 782 | **3.1x** | 11x |
| MetaWorldButtonPress | Meta-World | 480x480 | 1 | 216 | 2,767 (4,096) | 100% | 901 | **3.1x** | 13x |
| MetaWorldDisassemble | Meta-World | 480x480 | 1 | 213 | 2,837 (4,096) | 100% | 926 | **3.1x** | 13x |
| HumanoidBenchH1HandStair | HumanoidBench | 256x256 | 1 | 57 | 11,867 (4,096) | 124% | 3,898 | **3.0x** | 209x |
| MetaWorldPlateSlideSide | Meta-World | 480x480 | 1 | 217 | 2,578 (4,096) | 100% | 852 | **3.0x** | 12x |
| MetaWorldStickPush | Meta-World | 480x480 | 1 | 217 | 2,824 (4,096) | 100% | 947 | **3.0x** | 13x |
| MetaWorldStickPull | Meta-World | 480x480 | 1 | 214 | 2,831 (4,096) | 100% | 949 | **3.0x** | 13x |
| HumanoidBenchH1HandSpoon | HumanoidBench | 256x256 | 1 | 38 | 10,489 (1,024) | 100% | 3,538 | **3.0x** | 278x |
| MetaWorldDoorClose | Meta-World | 480x480 | 1 | 213 | 2,616 (4,096) | 100% | 897 | **2.9x** | 12x |
| MetaWorldDoorOpen | Meta-World | 480x480 | 1 | 214 | 2,681 (4,096) | 100% | 925 | **2.9x** | 13x |
| MetaWorldHandlePullSide | Meta-World | 480x480 | 1 | 218 | 2,713 (4,096) | 100% | 953 | **2.8x** | 12x |
| MetaWorldHandlePull | Meta-World | 480x480 | 1 | 216 | 2,702 (4,096) | 100% | 953 | **2.8x** | 12x |
| MetaWorldHandlePress | Meta-World | 480x480 | 1 | 165 | 2,697 (4,096) | 100% | 951 | **2.8x** | 16x |
| MetaWorldDoorLock | Meta-World | 480x480 | 1 | 212 | 2,616 (4,096) | 100% | 930 | **2.8x** | 12x |
| MetaWorldPlateSlideBack | Meta-World | 480x480 | 1 | 217 | 2,553 (4,096) | 100% | 950 | **2.7x** | 12x |
| RoboCasaCounterToSinkL45S33 | RoboCasa | 256x256 | 1 | 394 | 5,991 (4,096) | 100% | 2,904 | **2.1x** | 15x |
| RoboCasaCounterToSink | RoboCasa | 256x256 | 1 | 442 | 5,759 (4,096) | 100% | 2,990 | **1.9x** | 13x |
| MetaWorldWindowOpen | Meta-World | 480x480 | 1 | 212 | 1,172 (4,096) | 100% | 776 | **1.5x** | 6x |
| MetaWorldWindowClose | Meta-World | 480x480 | 1 | 214 | 1,176 (4,096) | 100% | 859 | **1.4x** | 6x |
| DMCBallInCupCatch | DM Control | 84x84 | 1 | 349 | 4,378 (4,096) | 120% | 7,480 | **0.6x** | 13x |

## By family

| family | envs | frame | default f/s | ours f/s | stored f/s/core | **cores replaced** |
|---|---|---|---|---|---|---|
| Griddly | 7 | 144x120… | 102–171 | 2,311,257–9,473,899 | 910–4,790 | **1,977.9–3,869.5x** |
| Overcooked | 5 | 192x224… | 27–1,189 | 3,596,911–8,055,812 | 2,511–3,974 | **1,057.7–2,777.8x** |
| Sokoban | 1 | 160x160 | 256 | 9,572,010 | 4,186 | **2,286.6x** |
| MiniGrid / BabyAI | 7 | 224x224… | 206–938 | 711,102–4,420,748 | 800–5,096 | **323.7–889.1x** |
| Jumanji | 8 | 200x200 | 4–82 | 403,573–1,282,433 | 2,829–3,984 | **120.9–351.3x** |
| MiniHack | 1 | 336x1264 | — | 244,515 | 721 | **339.0x** |
| Procgen | 4 | 64x64 | 873–974 | 316,300–1,576,433 | 7,511–10,815 | **29.2–193.7x** |
| PushT | 1 | 96x96 | 476 | 1,181,957 | 7,699 | **153.5x** |
| Craftax | 1 | 112x144 | — | 568,670 | 5,544 | **102.6x** |
| OGBench | 25 | 224x224 | 391–488 | 77,494–242,672 | 2,368–4,251 | **18.7–96.4x** |
| swm nav | 5 | 224x224… | 53–830 | 70,746–187,975 | 1,317–4,967 | **15.6–75.4x** |
| highway-env | 6 | 150x600… | — | 16,538–199,371 | 781–2,926 | **21.2–69.3x** |
| Crafter | 1 | 64x64 | — | 661,255 | 9,648 | **68.5x** |
| PGX | 4 | 240x240… | 21–97 | 77,336–107,858 | 2,336–3,130 | **24.9–43.3x** |
| D4RL PointMaze | 4 | 224x224 | 1,803–2,255 | 94,813 | 2,355–4,379 | **21.7–40.3x** |
| classic control | 5 | 400x600… | 283–452 | 31,111–38,396 | 1,101–1,441 | **22.8–29.1x** |
| DM Control | 16 | 84x84 | 272–366 | 4,378–124,436 | 7,194–10,514 | **0.6–16.9x** |
| LunarLander | 1 | 400x600 | 339 | 18,825 | 1,399 | **13.5x** |
| MuJoCo locomotion | 10 | 240x240 | 325–2,502 | 9,250–52,901 | 2,430–4,757 | **3.3–12.2x** |
| Fetch | 12 | 480x480 | 586–713 | 8,653–8,958 | 850–1,255 | **7.0–10.5x** |
| HumanoidBench | 17 | 256x256 | 38–60 | 10,489–19,496 | 1,916–3,920 | **3.0–8.2x** |
| RoboCasa | 4 | 256x256 | 394–546 | 5,688–13,326 | 1,833–2,990 | **1.9–4.9x** |
| Meta-World | 50 | 480x480 | 165–221 | 1,172–2,977 | 601–953 | **1.4–4.8x** |
| CarRacing | 1 | 96x96 | 132 | 26,927 | 8,118 | **3.3x** |

## Does the *load* half of load-and-decode matter?

Across 196 envs, reading a warm local JPEG file and decoding it runs at 0.56–1.91x the speed of decoding the same bytes already in memory (median 0.97x).
A stored-frame pipeline is decode-bound, not IO-bound, so the in-memory column is both the
favourable bar and very nearly the only one. This says nothing about a dataset that has to
cross a network — that is out of scope here and would do worse.

## Provenance and caveats

* Measured across 14 interpreters, since no single one can import every family.
* **Neither column travels unconditionally.** A row whose `cpu%` is near 0 is one kernel on a
  device the job owns and repeats across nodes to under a percent. A row at or above 100% is
  host-bound — the ray-cast families are, at 100-206% — and moves with the node and its load:
  DMCBallInCupCatch measured 3,664 f/s idle and 271-978 contended, a 13x spread. The
  official-renderer column is a single-threaded Python renderer sharing a node and has moved by
  more than 100x with load; read it as an order of magnitude.
* 9 envs have no `default` number because no reference package was installed in the
  interpreter that owns them. That is an absence, not a zero.
* One GPU against one CPU core is the right comparison for "store or regenerate on the node I
  have", and the wrong one for a per-watt or per-dollar argument.
