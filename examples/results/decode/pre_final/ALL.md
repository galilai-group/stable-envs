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
| GriddlySokoban | Griddly | 216x312 | 1 | 150 | 6,678,665 (16,384) | 101% | 2,130 | **3,136.1x** | 44,546x |
| GriddlyCookMePasta | Griddly | 264x336 | 1 | 132 | 5,891,181 (16,384) | 100% | 1,900 | **3,100.4x** | 44,672x |
| GriddlyButterflies | Griddly | 264x672 | 1 | 86 | 2,547,666 (4,096) | 100% | 949 | **2,683.7x** | 29,517x |
| GriddlyClusters | Griddly | 240x312 | 1 | 145 | 6,021,440 (16,384) | 101% | 2,364 | **2,547.0x** | 41,577x |
| Sokoban | Sokoban | 160x160 | 1 | 259 | 9,564,785 (16,384) | 100% | 3,843 | **2,488.8x** | 36,869x |
| GriddlySokoban2 | Griddly | 168x192 | 1 | 169 | 8,328,813 (16,384) | 101% | 3,600 | **2,313.8x** | 49,245x |
| OvercookedCoordRing | Overcooked | 224x224 | 1 | 1,211 | 6,977,194 (16,384) | 100% | 3,047 | **2,290.0x** | 5,762x |
| GriddlyLabyrinth | Griddly | 336x384 | 1 | 96 | 2,312,709 (16,384) | 101% | 1,044 | **2,215.8x** | 24,155x |
| OvercookedForcedCoord | Overcooked | 224x224 | 1 | 1,167 | 6,970,414 (16,384) | 100% | 3,170 | **2,198.9x** | 5,973x |
| OvercookedCrampedRoom | Overcooked | 192x224 | 1 | 475 | 8,065,016 (16,384) | 100% | 3,967 | **2,033.2x** | 16,995x |
| OvercookedCounterCircuit | Overcooked | 224x320 | 1 | 873 | 6,957,080 (16,384) | 100% | 3,651 | **1,905.6x** | 7,967x |
| GriddlyBait | Griddly | 144x120 | 1 | 161 | 9,504,280 (16,384) | 100% | 6,107 | **1,556.2x** | 58,853x |
| OvercookedAsymmAdvantages | Overcooked | 224x352 | 1 | 27 | 3,598,037 (16,384) | 100% | 2,560 | **1,405.7x** | 131,169x |
| MiniGridFourRooms | MiniGrid / BabyAI | 608x608 | 1 | 317 | 764,913 (4,096) | 100% | 894 | **855.8x** | 2,416x |
| MiniGridPOVFetch | MiniGrid / BabyAI | 224x224 | 1 | 644 | 4,704,072 (4,096) | 100% | 5,817 | **808.6x** | 7,305x |
| MiniGridLavaCrossing | MiniGrid / BabyAI | 288x288 | 1 | 1,241 | 1,323,276 (4,096) | 100% | 1,801 | **734.7x** | 1,066x |
| MiniGridKeyCorridor | MiniGrid / BabyAI | 320x320 | 1 | 301 | 1,300,694 (4,096) | 100% | 3,119 | **417.0x** | 4,317x |
| JumanjiSudokuVeryEasy | Jumanji | 200x200 | 1 | 10 | 1,283,774 (16,384) | 102% | 3,738 | **343.5x** | 123,334x |
| MiniGridEmpty | MiniGrid / BabyAI | 256x256 | 1 | 1,201 | 1,487,533 (4,096) | 100% | 4,390 | **338.9x** | 1,238x |
| JumanjiMinesweeper | Jumanji | 200x200 | 1 | 11 | 1,196,551 (16,384) | 100% | 3,628 | **329.8x** | 110,119x |
| BabyAIGoToLocal | MiniGrid / BabyAI | 256x256 | 1 | 549 | 1,367,171 (4,096) | 100% | 4,176 | **327.4x** | 2,492x |
| MiniGridDoorKey | MiniGrid / BabyAI | 256x256 | 1 | 568 | 1,365,120 (4,096) | 100% | 4,495 | **303.7x** | 2,404x |
| MiniHack | MiniHack | 336x1264 | 1 | — | 244,444 (4,096) | 100% | 807 | **303.1x** | — |
| JumanjiSudoku | Jumanji | 200x200 | 1 | 12 | 1,284,613 (16,384) | 100% | 4,482 | **286.6x** | 108,964x |
| ProcgenMaze | Procgen | 64x64 | 1 | 894 | 1,598,201 (4,096) | 100% | 7,164 | **223.1x** | 1,789x |
| JumanjiRubiksCube | Jumanji | 200x200 | 1 | 8 | 599,814 (16,384) | 100% | 3,572 | **167.9x** | 78,740x |
| JumanjiRubiksCubePartlyScrambled | Jumanji | 200x200 | 1 | 8 | 599,975 (16,384) | 100% | 3,662 | **163.8x** | 79,749x |
| JumanjiSokoban | Jumanji | 200x200 | 1 | 6 | 409,171 (16,384) | 100% | 2,823 | **144.9x** | 63,820x |
| JumanjiMaze | Jumanji | 200x200 | 1 | 83 | 571,077 (16,384) | 100% | 4,551 | **125.5x** | 6,863x |
| JumanjiConnector | Jumanji | 200x200 | 1 | 4 | 461,408 (16,384) | 100% | 3,722 | **124.0x** | 123,079x |
| PushT | PushT | 96x96 | 1 | 478 | 1,222,442 (4,096) | 100% | 10,089 | **121.2x** | 2,559x |
| ProcgenMiner | Procgen | 64x64 | 1 | 967 | 1,172,535 (4,096) | 100% | 10,413 | **112.6x** | 1,212x |
| CraftaxClassic | Craftax | 112x144 | 1 | — | 569,498 (4,096) | 100% | 5,484 | **103.9x** | — |
| PointmazeMediumStitch | OGBench | 224x224 | 2 | 488 | 241,610 (4,096) | 100% | 3,033 | **79.6x** | 495x |
| PointmazeGiantNavigate | OGBench | 224x224 | 2 | 410 | 241,486 (4,096) | 100% | 3,171 | **76.2x** | 589x |
| Highway | highway-env | 150x600 | 1 | — | 202,928 (4,096) | 100% | 2,703 | **75.1x** | — |
| SimplePointMaze | swm nav | 500x500 | 1 | 53 | 71,027 (4,096) | 100% | 1,090 | **65.2x** | 1,336x |
| PointmazeGiantStitch | OGBench | 224x224 | 2 | 420 | 240,921 (4,096) | 100% | 3,725 | **64.7x** | 574x |
| PointmazeTeleportStitch | OGBench | 224x224 | 2 | 456 | 241,624 (4,096) | 100% | 3,984 | **60.6x** | 530x |
| PointmazeTeleportNavigate | OGBench | 224x224 | 2 | 442 | 242,036 (4,096) | 100% | 3,996 | **60.6x** | 547x |
| PointmazeLargeStitch | OGBench | 224x224 | 2 | 454 | 242,023 (4,096) | 100% | 4,091 | **59.2x** | 533x |
| PointmazeLargeNavigate | OGBench | 224x224 | 2 | 438 | 240,802 (4,096) | 100% | 4,095 | **58.8x** | 550x |
| Merge | highway-env | 150x600 | 1 | — | 204,088 (4,096) | 100% | 3,558 | **57.4x** | — |
| PointmazeMediumNavigate | OGBench | 224x224 | 2 | 473 | 241,515 (4,096) | 100% | 4,223 | **57.2x** | 510x |
| AntmazeMediumNavigate | OGBench | 224x224 | 2 | 477 | 111,287 (4,096) | 329% | 2,033 | **54.7x** | 234x |
| AntmazeTeleportExplore | OGBench | 224x224 | 2 | 429 | 111,174 (4,096) | 328% | 2,038 | **54.6x** | 259x |
| PgxGo9x9 | PGX | 250x250 | 1 | 39 | 93,875 (4,096) | 100% | 1,791 | **52.4x** | 2,430x |
| Crafter | Crafter | 64x64 | 1 | — | 662,894 (4,096) | 100% | 12,840 | **51.6x** | — |
| PointMazeUMaze | D4RL PointMaze | 224x224 | 2 | 2,195 | 95,506 (4,096) | 100% | 2,235 | **42.7x** | 44x |
| Parking | highway-env | 300x600 | 1 | — | 63,372 (4,096) | 100% | 1,577 | **40.2x** | — |
| ProcgenHeist | Procgen | 64x64 | 1 | 943 | 321,124 (4,096) | 100% | 8,904 | **36.1x** | 340x |
| PointMazeOpen | D4RL PointMaze | 224x224 | 2 | 2,180 | 95,456 (4,096) | 100% | 2,650 | **36.0x** | 44x |
| Pendulum | classic control | 500x500 | 1 | 347 | 40,047 (4,096) | 100% | 1,121 | **35.7x** | 116x |
| PointMazeMedium | D4RL PointMaze | 224x224 | 2 | 2,124 | 95,424 (4,096) | 100% | 2,683 | **35.6x** | 45x |
| ProcgenChaser | Procgen | 64x64 | 1 | 871 | 352,101 (4,096) | 100% | 9,911 | **35.5x** | 404x |
| AntmazeLargeStitch | OGBench | 224x224 | 2 | 454 | 111,799 (4,096) | 330% | 3,233 | **34.6x** | 246x |
| AntmazeLargeNavigate | OGBench | 224x224 | 2 | 455 | 111,227 (4,096) | 330% | 3,304 | **33.7x** | 244x |
| PgxOthello | PGX | 270x270 | 1 | 20 | 81,939 (4,096) | 100% | 2,445 | **33.5x** | 4,010x |
| PgxTicTacToe | PGX | 240x240 | 1 | 89 | 109,184 (4,096) | 100% | 3,287 | **33.2x** | 1,220x |
| Piecewise | swm nav | 224x224 | 1 | 834 | 187,918 (4,096) | 100% | 5,826 | **32.3x** | 225x |
| CartPole | classic control | 400x600 | 1 | 535 | 39,019 (4,096) | 100% | 1,217 | **32.1x** | 73x |
| PointMazeLarge | D4RL PointMaze | 224x224 | 2 | 1,851 | 95,584 (4,096) | 100% | 3,018 | **31.7x** | 52x |
| Intersection | highway-env | 600x600 | 1 | — | 25,598 (1,024) | 100% | 818 | **31.3x** | — |
| AntmazeMediumStitch | OGBench | 224x224 | 2 | 476 | 113,903 (4,096) | 334% | 3,738 | **30.5x** | 239x |
| PiecewiseZones | swm nav | 224x224 | 1 | 385 | 140,794 (4,096) | 100% | 4,870 | **28.9x** | 366x |
| Racetrack | highway-env | 600x600 | 1 | — | 24,877 (1,024) | 100% | 863 | **28.8x** | — |
| HumanoidmazeLargeNavigate | OGBench | 224x224 | 2 | 448 | 95,491 (4,096) | 300% | 3,438 | **27.8x** | 213x |
| TwoRoomTarget | swm nav | 224x224 | 1 | 498 | 76,598 (4,096) | 100% | 2,795 | **27.4x** | 154x |
| HumanoidmazeMediumStitch | OGBench | 224x224 | 2 | 475 | 96,480 (4,096) | 304% | 3,523 | **27.4x** | 203x |
| AntmazeTeleportNavigate | OGBench | 224x224 | 2 | 441 | 110,930 (4,096) | 328% | 4,379 | **25.3x** | 252x |
| AntmazeTeleportStitch | OGBench | 224x224 | 2 | 442 | 110,719 (4,096) | 328% | 4,409 | **25.1x** | 251x |
| PgxConnectFour | PGX | 280x280 | 1 | 66 | 78,321 (4,096) | 100% | 3,129 | **25.0x** | 1,190x |
| HumanoidmazeLargeStitch | OGBench | 224x224 | 2 | 464 | 95,158 (4,096) | 298% | 3,852 | **24.7x** | 205x |
| AntmazeLargeExplore | OGBench | 224x224 | 2 | 443 | 111,916 (4,096) | 330% | 4,534 | **24.7x** | 253x |
| HumanoidmazeGiantStitch | OGBench | 224x224 | 2 | 414 | 92,482 (4,096) | 334% | 3,769 | **24.5x** | 223x |
| AntmazeMediumExplore | OGBench | 224x224 | 2 | 466 | 112,610 (4,096) | 332% | 4,722 | **23.8x** | 242x |
| AntmazeGiantStitch | OGBench | 224x224 | 2 | 410 | 108,193 (4,096) | 323% | 4,547 | **23.8x** | 264x |
| AntmazeGiantNavigate | OGBench | 224x224 | 2 | 405 | 107,976 (4,096) | 322% | 4,543 | **23.8x** | 267x |
| TwoRoom | swm nav | 224x224 | 1 | 685 | 116,673 (4,096) | 100% | 4,913 | **23.7x** | 170x |
| Acrobot | classic control | 500x500 | 1 | 472 | 34,046 (4,096) | 100% | 1,515 | **22.5x** | 72x |
| MountainCarContinuous | classic control | 400x600 | 1 | 516 | 33,500 (4,096) | 100% | 1,518 | **22.1x** | 65x |
| MountainCar | classic control | 400x600 | 1 | 504 | 33,526 (4,096) | 100% | 1,520 | **22.1x** | 67x |
| Roundabout | highway-env | 600x600 | 1 | — | 16,580 (1,024) | 100% | 789 | **21.0x** | — |
| HumanoidmazeMediumNavigate | OGBench | 224x224 | 2 | 461 | 96,376 (4,096) | 300% | 4,753 | **20.3x** | 209x |
| HumanoidmazeGiantNavigate | OGBench | 224x224 | 2 | 406 | 91,865 (4,096) | 294% | 4,629 | **19.8x** | 226x |
| DMCWalkerWalk | DM Control | 84x84 | 1 | 335 | 117,496 (4,096) | 100% | 6,763 | **17.4x** | 351x |
| LunarLander | LunarLander | 400x600 | 1 | 234 | 18,779 (4,096) | 100% | 1,257 | **14.9x** | 80x |
| DMCHumanoidStand | DM Control | 84x84 | 1 | 285 | 102,746 (4,096) | 100% | 7,112 | **14.4x** | 361x |
| DMCCheetahRun | DM Control | 84x84 | 1 | 352 | 126,997 (4,096) | 100% | 9,107 | **13.9x** | 360x |
| DMCHopperHop | DM Control | 84x84 | 1 | 333 | 126,504 (4,096) | 100% | 9,227 | **13.7x** | 380x |
| FetchPush | Fetch | 480x480 | 1 | 642 | 8,775 (4,096) | 100% | 740 | **11.9x** | 14x |
| FetchSlide | Fetch | 480x480 | 1 | 617 | 8,989 (4,096) | 100% | 786 | **11.4x** | 15x |
| FetchPushDict | Fetch | 480x480 | 1 | 744 | 8,781 (4,096) | 100% | 778 | **11.3x** | 12x |
| FetchPickAndPlaceDense | Fetch | 480x480 | 1 | 739 | 8,682 (4,096) | 100% | 797 | **10.9x** | 12x |
| HumanoidBenchH1HandStand | HumanoidBench | 256x256 | 1 | 38 | 17,112 (4,096) | 134% | 1,720 | **9.9x** | 451x |
| FetchReachDense | Fetch | 480x480 | 1 | 761 | 8,714 (4,096) | 100% | 955 | **9.1x** | 11x |
| DMCCartpoleBalance | DM Control | 84x84 | 2 | 351 | 51,123 (4,096) | 100% | 6,073 | **8.4x** | 146x |
| FetchReachDict | Fetch | 480x480 | 1 | 757 | 8,714 (4,096) | 100% | 1,085 | **8.0x** | 12x |
| FetchSlideDense | Fetch | 480x480 | 1 | 622 | 8,987 (4,096) | 100% | 1,228 | **7.3x** | 14x |
| DMCCartpoleSwingup | DM Control | 84x84 | 2 | 350 | 49,045 (4,096) | 100% | 6,802 | **7.2x** | 140x |
| HumanoidBenchH1BalanceSimple | HumanoidBench | 256x256 | 1 | 39 | 12,390 (4,096) | 125% | 1,744 | **7.1x** | 321x |
| HumanoidBenchH1HandStair | HumanoidBench | 256x256 | 1 | 57 | 11,710 (4,096) | 123% | 1,683 | **7.0x** | 206x |
| DMCPendulumSwingup | DM Control | 84x84 | 2 | 351 | 41,924 (4,096) | 100% | 6,196 | **6.8x** | 119x |
| HumanoidBenchH1Run | HumanoidBench | 256x256 | 1 | 39 | 19,027 (4,096) | 138% | 2,857 | **6.7x** | 491x |
| HumanoidBenchH1HandSitHard | HumanoidBench | 256x256 | 1 | 38 | 11,840 (4,096) | 124% | 1,782 | **6.6x** | 313x |
| FetchSlideDict | Fetch | 480x480 | 1 | 643 | 8,987 (4,096) | 100% | 1,368 | **6.6x** | 14x |
| FetchPushDense | Fetch | 480x480 | 1 | 753 | 8,777 (4,096) | 100% | 1,366 | **6.4x** | 12x |
| FetchReach | Fetch | 480x480 | 1 | 674 | 8,710 (4,096) | 100% | 1,389 | **6.3x** | 13x |
| FetchPickAndPlace | Fetch | 480x480 | 1 | 658 | 8,686 (4,096) | 100% | 1,401 | **6.2x** | 13x |
| DMCSwimmerSwimmer6 | DM Control | 84x84 | 1 | 269 | 65,533 (4,096) | 100% | 10,573 | **6.2x** | 243x |
| HumanoidBenchH1HandSpoon | HumanoidBench | 256x256 | 1 | 38 | 10,375 (1,024) | 100% | 1,674 | **6.2x** | 275x |
| FetchPickAndPlaceDict | Fetch | 480x480 | 1 | 758 | 8,687 (4,096) | 100% | 1,412 | **6.2x** | 11x |
| HumanoidBenchH1Stand | HumanoidBench | 256x256 | 1 | 39 | 19,025 (4,096) | 138% | 3,131 | **6.1x** | 491x |
| HumanoidBenchH1HandRun | HumanoidBench | 256x256 | 1 | 38 | 17,110 (4,096) | 134% | 2,839 | **6.0x** | 452x |
| HumanoidBenchH1HandBalanceSimple | HumanoidBench | 256x256 | 1 | 38 | 11,121 (4,096) | 122% | 1,847 | **6.0x** | 294x |
| DMCPointMassEasy | DM Control | 84x84 | 2 | 367 | 37,536 (4,096) | 100% | 6,393 | **5.9x** | 102x |
| HumanoidBenchH1Stair | HumanoidBench | 256x256 | 1 | 59 | 13,002 (4,096) | 126% | 2,278 | **5.7x** | 221x |
| MujocoSwimmer | MuJoCo locomotion | 240x240 | 1 | 316 | 10,449 (4,096) | 121% | 1,865 | **5.6x** | 33x |
| DMCFingerSpin | DM Control | 84x84 | 2 | 364 | 35,819 (4,096) | 601% | 6,527 | **5.5x** | 98x |
| HumanoidBenchH1SitHard | HumanoidBench | 256x256 | 1 | 39 | 13,233 (4,096) | 126% | 2,456 | **5.4x** | 343x |
| HumanoidBenchH1Walk | HumanoidBench | 256x256 | 1 | 39 | 19,023 (4,096) | 138% | 4,340 | **4.4x** | 491x |
| RoboCasaCounterToSinkWrist | RoboCasa | 256x256 | 1 | 562 | 13,311 (1,024) | 100% | 3,051 | **4.4x** | 24x |
| HumanoidBenchH1HandSitSimple | HumanoidBench | 256x256 | 1 | 38 | 11,967 (4,096) | 124% | 2,750 | **4.4x** | 316x |
| HumanoidBenchH1SitSimple | HumanoidBench | 256x256 | 1 | 39 | 13,194 (4,096) | 126% | 3,148 | **4.2x** | 342x |
| DMCFingerTurnHard | DM Control | 84x84 | 2 | 351 | 26,531 (4,096) | 471% | 6,425 | **4.1x** | 76x |
| MujocoHalfCheetah | MuJoCo locomotion | 240x240 | 1 | 335 | 10,512 (4,096) | 120% | 2,563 | **4.1x** | 31x |
| MetaWorldPegInsertSide | Meta-World | 480x480 | 1 | 230 | 2,838 (4,096) | 100% | 693 | **4.1x** | 12x |
| HumanoidBenchH1HandWalk | HumanoidBench | 256x256 | 1 | 38 | 17,122 (4,096) | 134% | 4,260 | **4.0x** | 452x |
| MetaWorldHandInsert | Meta-World | 480x480 | 1 | 228 | 2,966 (4,096) | 100% | 758 | **3.9x** | 13x |
| HumanoidBenchH1Slide | HumanoidBench | 256x256 | 1 | 60 | 16,837 (4,096) | 134% | 4,323 | **3.9x** | 281x |
| MujocoHopper | MuJoCo locomotion | 240x240 | 1 | 257 | 10,527 (4,096) | 120% | 2,719 | **3.9x** | 41x |
| MetaWorldBinPicking | Meta-World | 480x480 | 1 | 228 | 2,896 (4,096) | 100% | 766 | **3.8x** | 13x |
| MujocoHumanoid | MuJoCo locomotion | 240x240 | 1 | 296 | 9,044 (4,096) | 144% | 2,429 | **3.7x** | 31x |
| MetaWorldDisassemble | Meta-World | 480x480 | 1 | 225 | 2,846 (4,096) | 100% | 767 | **3.7x** | 13x |
| MetaWorldDrawerOpen | Meta-World | 480x480 | 1 | 230 | 2,854 (4,096) | 100% | 769 | **3.7x** | 12x |
| MetaWorldDialTurn | Meta-World | 480x480 | 1 | 228 | 2,939 (4,096) | 100% | 795 | **3.7x** | 13x |
| MetaWorldFaucetClose | Meta-World | 480x480 | 1 | 226 | 2,848 (4,096) | 100% | 771 | **3.7x** | 13x |
| MetaWorldFaucetOpen | Meta-World | 480x480 | 1 | 226 | 2,849 (4,096) | 100% | 781 | **3.6x** | 13x |
| CarRacing | CarRacing | 96x96 | 1 | 130 | 26,833 (4,096) | 100% | 7,384 | **3.6x** | 206x |
| MetaWorldButtonPress | Meta-World | 480x480 | 1 | 230 | 2,780 (4,096) | 100% | 768 | **3.6x** | 12x |
| MetaWorldAssembly | Meta-World | 480x480 | 1 | 227 | 2,890 (4,096) | 100% | 811 | **3.6x** | 13x |
| DMCFingerTurnEasy | DM Control | 84x84 | 2 | 359 | 26,134 (4,096) | 465% | 7,332 | **3.6x** | 73x |
| MetaWorldCoffeePush | Meta-World | 480x480 | 1 | 228 | 2,754 (4,096) | 100% | 773 | **3.6x** | 12x |
| MetaWorldStickPush | Meta-World | 480x480 | 1 | 228 | 2,833 (4,096) | 100% | 797 | **3.6x** | 12x |
| MetaWorldHandlePressSide | Meta-World | 480x480 | 1 | 224 | 2,715 (4,096) | 100% | 766 | **3.5x** | 12x |
| HumanoidBenchH1HandSlide | HumanoidBench | 256x256 | 1 | 58 | 15,255 (4,096) | 131% | 4,343 | **3.5x** | 264x |
| DMCAcrobotSwingup | DM Control | 84x84 | 2 | 362 | 35,195 (4,096) | 100% | 10,104 | **3.5x** | 97x |
| MetaWorldPickOutOfHole | Meta-World | 480x480 | 1 | 222 | 2,858 (4,096) | 100% | 823 | **3.5x** | 13x |
| MetaWorldDoorUnlock | Meta-World | 480x480 | 1 | 226 | 2,624 (4,096) | 100% | 764 | **3.4x** | 12x |
| MetaWorldDoorLock | Meta-World | 480x480 | 1 | 226 | 2,621 (4,096) | 100% | 786 | **3.3x** | 12x |
| MetaWorldReachWall | Meta-World | 480x480 | 1 | 229 | 2,949 (4,096) | 100% | 890 | **3.3x** | 13x |
| DMCReacherEasy | DM Control | 84x84 | 2 | 357 | 24,238 (4,096) | 100% | 7,366 | **3.3x** | 68x |
| MetaWorldPushBack | Meta-World | 480x480 | 1 | 231 | 2,971 (4,096) | 100% | 906 | **3.3x** | 13x |
| DMCReacherHard | DM Control | 84x84 | 2 | 350 | 24,892 (4,096) | 100% | 7,658 | **3.3x** | 71x |
| MetaWorldPickPlaceWall | Meta-World | 480x480 | 1 | 229 | 2,949 (4,096) | 100% | 908 | **3.2x** | 13x |
| MujocoReacher | MuJoCo locomotion | 240x240 | 2 | 2,240 | 11,304 (4,096) | 100% | 3,540 | **3.2x** | 5x |
| MetaWorldBasketball | Meta-World | 480x480 | 1 | 230 | 2,818 (4,096) | 100% | 886 | **3.2x** | 12x |
| MetaWorldButtonPressWall | Meta-World | 480x480 | 1 | 228 | 2,755 (4,096) | 100% | 898 | **3.1x** | 12x |
| MetaWorldButtonPressTopdownWall | Meta-World | 480x480 | 1 | 226 | 2,799 (4,096) | 100% | 915 | **3.1x** | 12x |
| MetaWorldPickPlace | Meta-World | 480x480 | 1 | 229 | 2,968 (4,096) | 100% | 989 | **3.0x** | 13x |
| RoboCasaCounterToSinkL45S33 | RoboCasa | 256x256 | 1 | 406 | 5,907 (4,096) | 100% | 1,986 | **3.0x** | 15x |
| MetaWorldButtonPressTopdown | Meta-World | 480x480 | 1 | 229 | 2,818 (4,096) | 100% | 967 | **2.9x** | 12x |
| MetaWorldPlateSlide | Meta-World | 480x480 | 1 | 229 | 2,576 (4,096) | 100% | 886 | **2.9x** | 11x |
| MujocoWalker2d | MuJoCo locomotion | 240x240 | 1 | 370 | 10,544 (4,096) | 122% | 3,641 | **2.9x** | 28x |
| RoboCasaCounterToSinkRight | RoboCasa | 256x256 | 1 | 456 | 5,608 (4,096) | 100% | 1,940 | **2.9x** | 12x |
| MetaWorldSweepInto | Meta-World | 480x480 | 1 | 230 | 2,982 (4,096) | 100% | 1,040 | **2.9x** | 13x |
| MetaWorldPushWall | Meta-World | 480x480 | 1 | 227 | 2,951 (4,096) | 100% | 1,040 | **2.8x** | 13x |
| MetaWorldSweep | Meta-World | 480x480 | 1 | 230 | 2,971 (4,096) | 100% | 1,049 | **2.8x** | 13x |
| MetaWorldLeverPull | Meta-World | 480x480 | 1 | 230 | 2,778 (4,096) | 100% | 985 | **2.8x** | 12x |
| MetaWorldPush | Meta-World | 480x480 | 1 | 230 | 2,969 (4,096) | 100% | 1,057 | **2.8x** | 13x |
| MetaWorldReach | Meta-World | 480x480 | 1 | 229 | 2,969 (4,096) | 100% | 1,058 | **2.8x** | 13x |
| MujocoAnt | MuJoCo locomotion | 240x240 | 1 | 207 | 9,774 (4,096) | 129% | 3,515 | **2.8x** | 47x |
| MetaWorldShelfPlace | Meta-World | 480x480 | 1 | 229 | 2,748 (4,096) | 100% | 1,003 | **2.7x** | 12x |
| MetaWorldBoxClose | Meta-World | 480x480 | 1 | 229 | 2,869 (4,096) | 100% | 1,049 | **2.7x** | 13x |
| MetaWorldCoffeeButton | Meta-World | 480x480 | 1 | 228 | 2,759 (4,096) | 100% | 1,015 | **2.7x** | 12x |
| MetaWorldDrawerClose | Meta-World | 480x480 | 1 | 231 | 2,837 (4,096) | 100% | 1,045 | **2.7x** | 12x |
| MetaWorldStickPull | Meta-World | 480x480 | 1 | 226 | 2,839 (4,096) | 100% | 1,046 | **2.7x** | 13x |
| MetaWorldCoffeePull | Meta-World | 480x480 | 1 | 226 | 2,761 (4,096) | 100% | 1,018 | **2.7x** | 12x |
| MetaWorldHammer | Meta-World | 480x480 | 1 | 226 | 2,785 (4,096) | 100% | 1,029 | **2.7x** | 12x |
| MetaWorldHandlePullSide | Meta-World | 480x480 | 1 | 227 | 2,720 (4,096) | 100% | 1,016 | **2.7x** | 12x |
| MetaWorldPegUnplugSide | Meta-World | 480x480 | 1 | 221 | 2,837 (4,096) | 100% | 1,063 | **2.7x** | 13x |
| MetaWorldHandlePull | Meta-World | 480x480 | 1 | 227 | 2,708 (4,096) | 100% | 1,015 | **2.7x** | 12x |
| MetaWorldDoorClose | Meta-World | 480x480 | 1 | 229 | 2,622 (4,096) | 100% | 1,001 | **2.6x** | 11x |
| MetaWorldHandlePress | Meta-World | 480x480 | 1 | 228 | 2,704 (4,096) | 100% | 1,042 | **2.6x** | 12x |
| MetaWorldSoccer | Meta-World | 480x480 | 1 | 231 | 2,413 (4,096) | 100% | 931 | **2.6x** | 10x |
| MetaWorldDoorOpen | Meta-World | 480x480 | 1 | 226 | 2,689 (4,096) | 100% | 1,045 | **2.6x** | 12x |
| MetaWorldPlateSlideSide | Meta-World | 480x480 | 1 | 228 | 2,584 (4,096) | 100% | 1,006 | **2.6x** | 11x |
| MetaWorldPlateSlideBackSide | Meta-World | 480x480 | 1 | 227 | 2,596 (4,096) | 100% | 1,015 | **2.6x** | 11x |
| MetaWorldPlateSlideBack | Meta-World | 480x480 | 1 | 229 | 2,561 (4,096) | 100% | 1,039 | **2.5x** | 11x |
| RoboCasaCounterToSink | RoboCasa | 256x256 | 1 | 450 | 5,680 (4,096) | 100% | 2,403 | **2.4x** | 13x |
| MujocoInvertedPendulum | MuJoCo locomotion | 240x240 | 2 | 2,946 | 11,012 (4,096) | 100% | 5,259 | **2.1x** | 4x |
| MujocoInvertedDoublePendulum | MuJoCo locomotion | 240x240 | 2 | 2,233 | 11,250 (4,096) | 100% | 5,480 | **2.1x** | 5x |
| MujocoPusher | MuJoCo locomotion | 240x240 | 2 | 476 | 5,690 (4,096) | 100% | 3,733 | **1.5x** | 12x |
| MetaWorldWindowClose | Meta-World | 480x480 | 1 | 223 | 1,177 (4,096) | 100% | 870 | **1.4x** | 5x |
| MetaWorldWindowOpen | Meta-World | 480x480 | 1 | 221 | 1,173 (4,096) | 100% | 1,003 | **1.2x** | 5x |
| DMCBallInCupCatch | DM Control | 84x84 | 2 | 345 | 4,135 (4,096) | 161% | 7,634 | **0.5x** | 12x |

## By family

| family | envs | frame | default f/s | ours f/s | stored f/s/core | **cores replaced** |
|---|---|---|---|---|---|---|
| Griddly | 7 | 144x120… | 86–169 | 2,312,709–9,504,280 | 949–6,107 | **1,556.2–3,136.1x** |
| Sokoban | 1 | 160x160 | 259 | 9,564,785 | 3,843 | **2,488.8x** |
| Overcooked | 5 | 192x224… | 27–1,211 | 3,598,037–8,065,016 | 2,560–3,967 | **1,405.7–2,290.0x** |
| MiniGrid / BabyAI | 7 | 224x224… | 301–1,241 | 764,913–4,704,072 | 894–5,817 | **303.7–855.8x** |
| Jumanji | 8 | 200x200 | 4–83 | 409,171–1,284,613 | 2,823–4,551 | **124.0–343.5x** |
| MiniHack | 1 | 336x1264 | — | 244,444 | 807 | **303.1x** |
| Procgen | 4 | 64x64 | 871–967 | 321,124–1,598,201 | 7,164–10,413 | **35.5–223.1x** |
| PushT | 1 | 96x96 | 478 | 1,222,442 | 10,089 | **121.2x** |
| Craftax | 1 | 112x144 | — | 569,498 | 5,484 | **103.9x** |
| OGBench | 25 | 224x224 | 405–488 | 91,865–242,036 | 2,033–4,753 | **19.8–79.6x** |
| highway-env | 6 | 150x600… | — | 16,580–204,088 | 789–3,558 | **21.0–75.1x** |
| swm nav | 5 | 224x224… | 53–834 | 71,027–187,918 | 1,090–5,826 | **23.7–65.2x** |
| PGX | 4 | 240x240… | 20–89 | 78,321–109,184 | 1,791–3,287 | **25.0–52.4x** |
| Crafter | 1 | 64x64 | — | 662,894 | 12,840 | **51.6x** |
| D4RL PointMaze | 4 | 224x224 | 1,851–2,195 | 95,424 | 2,235–3,018 | **31.7–42.7x** |
| classic control | 5 | 400x600… | 347–535 | 33,500–40,047 | 1,121–1,520 | **22.1–35.7x** |
| DM Control | 16 | 84x84 | 269–367 | 4,135–126,997 | 6,073–10,573 | **0.5–17.4x** |
| LunarLander | 1 | 400x600 | 234 | 18,779 | 1,257 | **14.9x** |
| Fetch | 12 | 480x480 | 617–761 | 8,682–8,989 | 740–1,412 | **6.2–11.9x** |
| HumanoidBench | 17 | 256x256 | 38–60 | 10,375–19,027 | 1,674–4,343 | **3.5–9.9x** |
| MuJoCo locomotion | 10 | 240x240 | 207–2,946 | 5,690–11,304 | 1,865–5,480 | **1.5–5.6x** |
| RoboCasa | 4 | 256x256 | 406–562 | 5,608–13,311 | 1,940–3,051 | **2.4–4.4x** |
| Meta-World | 50 | 480x480 | 221–231 | 1,173–2,982 | 693–1,063 | **1.2–4.1x** |
| CarRacing | 1 | 96x96 | 130 | 26,833 | 7,384 | **3.6x** |

## Does the *load* half of load-and-decode matter?

Across 196 envs, reading a warm local JPEG file and decoding it runs at 0.44–2.38x the speed of decoding the same bytes already in memory (median 0.97x).
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
