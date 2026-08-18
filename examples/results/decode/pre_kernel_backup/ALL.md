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
| GriddlySokoban | Griddly | 216x312 | ? | 145 | 6,661,795 (16,384) | ? | 1,826 | **3,647.9x** | 45,937x |
| GriddlyCookMePasta | Griddly | 264x336 | ? | 142 | 5,880,109 (16,384) | ? | 1,913 | **3,073.4x** | 41,371x |
| GriddlyClusters | Griddly | 240x312 | ? | 133 | 6,048,008 (16,384) | ? | 2,036 | **2,970.0x** | 45,416x |
| GriddlyButterflies | Griddly | 264x672 | ? | 99 | 2,521,804 (4,096) | ? | 851 | **2,964.4x** | 25,419x |
| Sokoban | Sokoban | 160x160 | ? | 253 | 9,571,978 (16,384) | ? | 3,983 | **2,403.2x** | 37,785x |
| GriddlyLabyrinth | Griddly | 336x384 | ? | 122 | 2,307,114 (16,384) | ? | 1,037 | **2,224.8x** | 18,907x |
| GriddlySokoban2 | Griddly | 168x192 | ? | 101 | 8,292,010 (16,384) | ? | 3,770 | **2,199.3x** | 82,365x |
| OvercookedCrampedRoom | Overcooked | 192x224 | ? | 691 | 8,145,368 (16,384) | ? | 4,378 | **1,860.7x** | 11,795x |
| GriddlyBait | Griddly | 144x120 | ? | 115 | 9,438,816 (16,384) | ? | 5,276 | **1,789.0x** | 81,764x |
| OvercookedCounterCircuit | Overcooked | 224x320 | ? | 1,160 | 7,039,500 (16,384) | ? | 4,094 | **1,719.3x** | 6,071x |
| OvercookedForcedCoord | Overcooked | 224x224 | ? | 1,548 | 7,042,868 (16,384) | ? | 4,397 | **1,601.9x** | 4,550x |
| OvercookedCoordRing | Overcooked | 224x224 | ? | 1,595 | 7,045,754 (16,384) | ? | 4,399 | **1,601.7x** | 4,419x |
| OvercookedAsymmAdvantages | Overcooked | 224x352 | ? | 41 | 3,615,605 (16,384) | ? | 2,914 | **1,241.0x** | 87,586x |
| MiniGridPOVFetch | MiniGrid / BabyAI | 224x224 | ? | 439 | 4,479,313 (4,096) | ? | 3,654 | **1,225.8x** | 10,200x |
| MiniGridFourRooms | MiniGrid / BabyAI | 608x608 | ? | 117 | 711,991 (4,096) | ? | 608 | **1,171.6x** | 6,102x |
| MiniGridKeyCorridor | MiniGrid / BabyAI | 320x320 | ? | 206 | 1,157,515 (4,096) | ? | 1,581 | **732.2x** | 5,613x |
| MiniGridDoorKey | MiniGrid / BabyAI | 256x256 | ? | 239 | 1,210,788 (4,096) | ? | 1,987 | **609.5x** | 5,075x |
| BabyAIGoToLocal | MiniGrid / BabyAI | 256x256 | ? | 291 | 1,413,969 (4,096) | ? | 3,086 | **458.2x** | 4,864x |
| MiniGridLavaCrossing | MiniGrid / BabyAI | 288x288 | ? | 942 | 1,183,728 (4,096) | ? | 2,894 | **409.0x** | 1,256x |
| MiniHack | MiniHack | 336x1264 | ? | — | 244,456 (4,096) | ? | 698 | **350.0x** | — |
| JumanjiSudoku | Jumanji | 200x200 | ? | 10 | 1,286,366 (16,384) | ? | 3,780 | **340.4x** | 122,513x |
| JumanjiSudokuVeryEasy | Jumanji | 200x200 | ? | 9 | 1,286,308 (16,384) | ? | 3,813 | **337.4x** | 139,806x |
| MiniGridEmpty | MiniGrid / BabyAI | 256x256 | ? | 852 | 1,322,277 (4,096) | ? | 4,011 | **329.7x** | 1,551x |
| JumanjiMinesweeper | Jumanji | 200x200 | ? | 10 | 1,199,174 (16,384) | ? | 4,099 | **292.5x** | 121,545x |
| ProcgenMaze | Procgen | 64x64 | ? | 922 | 1,588,402 (4,096) | ? | 8,344 | **190.4x** | 1,723x |
| JumanjiRubiksCube | Jumanji | 200x200 | ? | 8 | 600,181 (16,384) | ? | 3,754 | **159.9x** | 77,933x |
| JumanjiRubiksCubePartlyScrambled | Jumanji | 200x200 | ? | 8 | 600,069 (16,384) | ? | 3,889 | **154.3x** | 78,574x |
| JumanjiSokoban | Jumanji | 200x200 | ? | 6 | 409,277 (16,384) | ? | 3,086 | **132.6x** | 65,091x |
| JumanjiMaze | Jumanji | 200x200 | ? | 88 | 571,134 (16,384) | ? | 4,310 | **132.5x** | 6,455x |
| PushT | PushT | 96x96 | ? | 472 | 1,256,557 (4,096) | ? | 10,259 | **122.5x** | 2,664x |
| ProcgenMiner | Procgen | 64x64 | ? | 975 | 1,157,231 (4,096) | ? | 10,655 | **108.6x** | 1,187x |
| JumanjiConnector | Jumanji | 200x200 | ? | 3 | 461,254 (16,384) | ? | 4,387 | **105.2x** | 145,393x |
| PointmazeGiantNavigate | OGBench | 224x224 | 2 | 410 | 205,708 (16,384) | 150% | 2,197 | **93.6x** | 501x |
| PointmazeTeleportStitch | OGBench | 224x224 | 2 | 459 | 206,329 (16,384) | 150% | 2,264 | **91.1x** | 449x |
| Piecewise | swm nav | 224x224 | ? | 772 | 187,629 (4,096) | ? | 2,163 | **86.7x** | 243x |
| Crafter | Crafter | 64x64 | ? | — | 665,836 (4,096) | ? | 8,186 | **81.3x** | — |
| CraftaxClassic | Craftax | 112x144 | ? | — | 569,934 (4,096) | ? | 7,039 | **81.0x** | — |
| PointmazeLargeStitch | OGBench | 224x224 | 2 | 455 | 206,740 (16,384) | 150% | 3,138 | **65.9x** | 455x |
| SimplePointMaze | swm nav | 500x500 | ? | 52 | 71,140 (4,096) | ? | 1,083 | **65.7x** | 1,356x |
| PointmazeLargeNavigate | OGBench | 224x224 | 2 | 440 | 205,912 (16,384) | 150% | 3,525 | **58.4x** | 468x |
| PointmazeTeleportNavigate | OGBench | 224x224 | 2 | 443 | 206,585 (16,384) | 149% | 3,968 | **52.1x** | 466x |
| PointmazeGiantStitch | OGBench | 224x224 | 2 | 421 | 205,342 (16,384) | 149% | 4,111 | **50.0x** | 487x |
| PointmazeMediumNavigate | OGBench | 224x224 | 2 | 473 | 206,135 (16,384) | 149% | 4,237 | **48.6x** | 436x |
| PointmazeMediumStitch | OGBench | 224x224 | 2 | 491 | 206,002 (16,384) | 150% | 4,249 | **48.5x** | 420x |
| PgxGo9x9 | PGX | 250x250 | ? | 45 | 93,958 (4,096) | ? | 2,055 | **45.7x** | 2,067x |
| ProcgenChaser | Procgen | 64x64 | ? | 899 | 347,799 (4,096) | ? | 9,119 | **38.1x** | 387x |
| Pendulum | classic control | 500x500 | ? | 312 | 39,843 (4,096) | ? | 1,120 | **35.6x** | 128x |
| PgxTicTacToe | PGX | 240x240 | ? | 108 | 109,484 (4,096) | ? | 3,160 | **34.6x** | 1,010x |
| PiecewiseZones | swm nav | 224x224 | ? | 500 | 140,483 (4,096) | ? | 4,144 | **33.9x** | 281x |
| TwoRoom | swm nav | 224x224 | ? | 696 | 116,507 (4,096) | ? | 3,655 | **31.9x** | 167x |
| AntmazeGiantNavigate | OGBench | 224x224 | 2 | 356 | 60,321 (16,384) | 132% | 1,940 | **31.1x** | 169x |
| AntmazeTeleportStitch | OGBench | 224x224 | 2 | 420 | 62,778 (16,384) | 133% | 2,035 | **30.9x** | 149x |
| PointMazeLarge | D4RL PointMaze | 224x224 | 2 | 1,846 | 88,303 (16,384) | 163% | 2,919 | **30.3x** | 48x |
| AntmazeTeleportExplore | OGBench | 224x224 | 2 | 371 | 56,706 (16,384) | 131% | 1,893 | **30.0x** | 153x |
| AntmazeLargeNavigate | OGBench | 224x224 | 2 | 440 | 59,252 (16,384) | 132% | 1,980 | **29.9x** | 135x |
| MountainCar | classic control | 400x600 | ? | 485 | 33,216 (4,096) | ? | 1,150 | **28.9x** | 69x |
| AntmazeLargeExplore | OGBench | 224x224 | 2 | 380 | 58,159 (16,384) | 131% | 2,028 | **28.7x** | 153x |
| AntmazeGiantStitch | OGBench | 224x224 | 2 | 390 | 54,223 (4,096) | 204% | 1,902 | **28.5x** | 139x |
| Acrobot | classic control | 500x500 | ? | 249 | 34,288 (4,096) | ? | 1,203 | **28.5x** | 138x |
| ProcgenHeist | Procgen | 64x64 | ? | 970 | 313,372 (4,096) | ? | 11,055 | **28.3x** | 323x |
| PgxOthello | PGX | 270x270 | ? | 21 | 82,027 (4,096) | ? | 3,139 | **26.1x** | 3,931x |
| PgxConnectFour | PGX | 280x280 | ? | 76 | 78,322 (4,096) | ? | 3,130 | **25.0x** | 1,034x |
| HumanoidmazeLargeStitch | OGBench | 224x224 | 2 | 448 | 49,488 (4,096) | 197% | 1,986 | **24.9x** | 110x |
| CartPole | classic control | 400x600 | ? | 266 | 39,393 (4,096) | ? | 1,626 | **24.2x** | 148x |
| HumanoidmazeLargeNavigate | OGBench | 224x224 | 2 | 119 | 52,286 (16,384) | 129% | 2,168 | **24.1x** | 438x |
| DMCCheetahRun | DM Control | 84x84 | ? | 348 | 140,391 (4,096) | ? | 6,110 | **23.0x** | 404x |
| MountainCarContinuous | classic control | 400x600 | ? | 541 | 33,236 (4,096) | ? | 1,493 | **22.3x** | 61x |
| PointMazeMedium | D4RL PointMaze | 224x224 | 2 | 2,045 | 88,935 (16,384) | 163% | 4,095 | **21.7x** | 43x |
| AntmazeTeleportNavigate | OGBench | 224x224 | 2 | 421 | 62,599 (16,384) | 133% | 2,898 | **21.6x** | 149x |
| AntmazeMediumStitch | OGBench | 224x224 | 2 | 462 | 63,620 (16,384) | 134% | 3,058 | **20.8x** | 138x |
| AntmazeMediumNavigate | OGBench | 224x224 | 2 | 463 | 63,461 (16,384) | 134% | 3,101 | **20.5x** | 137x |
| PointMazeUMaze | D4RL PointMaze | 224x224 | 2 | 2,227 | 88,489 (16,384) | 164% | 4,383 | **20.2x** | 40x |
| PointMazeOpen | D4RL PointMaze | 224x224 | 2 | 2,238 | 88,570 (16,384) | 163% | 4,403 | **20.1x** | 40x |
| HumanoidmazeGiantNavigate | OGBench | 224x224 | 2 | 117 | 53,307 (16,384) | 130% | 2,823 | **18.9x** | 454x |
| DMCHopperHop | DM Control | 84x84 | ? | 332 | 139,716 (4,096) | ? | 7,444 | **18.8x** | 421x |
| LunarLander | LunarLander | 400x600 | ? | 336 | 18,786 (4,096) | ? | 1,165 | **16.1x** | 56x |
| TwoRoomTarget | swm nav | 224x224 | ? | 505 | 76,474 (4,096) | ? | 4,985 | **15.3x** | 151x |
| AntmazeMediumExplore | OGBench | 224x224 | 2 | 409 | 59,909 (16,384) | 132% | 4,188 | **14.3x** | 146x |
| DMCHumanoidStand | DM Control | 84x84 | ? | 282 | 113,419 (4,096) | ? | 8,356 | **13.6x** | 403x |
| AntmazeLargeStitch | OGBench | 224x224 | 2 | 437 | 54,987 (4,096) | 206% | 4,085 | **13.5x** | 126x |
| DMCWalkerWalk | DM Control | 84x84 | ? | 335 | 130,347 (4,096) | ? | 10,291 | **12.7x** | 389x |
| HumanoidmazeMediumNavigate | OGBench | 224x224 | 2 | 82 | 51,716 (16,384) | 129% | 4,204 | **12.3x** | 630x |
| HumanoidmazeMediumStitch | OGBench | 224x224 | 2 | 457 | 50,617 (16,384) | 129% | 4,248 | **11.9x** | 111x |
| HumanoidmazeGiantStitch | OGBench | 224x224 | 2 | 397 | 48,978 (16,384) | 128% | 4,132 | **11.9x** | 123x |
| DMCSwimmerSwimmer6 | DM Control | 84x84 | ? | 272 | 73,358 (4,096) | ? | 9,087 | **8.1x** | 270x |
| Merge | highway-env | 150x600 | ? | — | 20,389 (4,096) | ? | 2,826 | **7.2x** | — |
| FetchSlideDense | Fetch | 480x480 | ? | 354 | 6,164 (4,096) | ? | 896 | **6.9x** | 17x |
| HumanoidBenchH1Walk | HumanoidBench | 256x256 | ? | 39 | 16,749 (256) | ? | 2,460 | **6.8x** | 433x |
| FetchSlideDict | Fetch | 480x480 | ? | 366 | 6,162 (1,024) | ? | 925 | **6.7x** | 17x |
| HumanoidBenchH1Run | HumanoidBench | 256x256 | ? | 38 | 16,835 (4,096) | ? | 2,628 | **6.4x** | 446x |
| HumanoidBenchH1HandStand | HumanoidBench | 256x256 | ? | 35 | 14,429 (256) | ? | 2,558 | **5.6x** | 408x |
| HumanoidBenchH1HandSlide | HumanoidBench | 256x256 | ? | 57 | 13,084 (64) | ? | 2,358 | **5.5x** | 228x |
| FetchPickAndPlaceDense | Fetch | 480x480 | ? | 363 | 6,010 (4,096) | ? | 1,111 | **5.4x** | 17x |
| FetchReachDense | Fetch | 480x480 | ? | 403 | 6,050 (1,024) | ? | 1,120 | **5.4x** | 15x |
| Highway | highway-env | 150x600 | ? | — | 17,214 (4,096) | ? | 3,205 | **5.4x** | — |
| FetchReach | Fetch | 480x480 | ? | 354 | 6,048 (4,096) | ? | 1,134 | **5.3x** | 17x |
| HumanoidBenchH1Slide | HumanoidBench | 256x256 | ? | 59 | 14,993 (4,096) | ? | 3,180 | **4.7x** | 252x |
| FetchSlide | Fetch | 480x480 | ? | 323 | 6,150 (1,024) | ? | 1,349 | **4.6x** | 19x |
| HumanoidBenchH1HandSitHard | HumanoidBench | 256x256 | ? | 35 | 10,488 (256) | ? | 2,347 | **4.5x** | 301x |
| FetchPushDense | Fetch | 480x480 | ? | 332 | 6,051 (4,096) | ? | 1,356 | **4.5x** | 18x |
| FetchReachDict | Fetch | 480x480 | ? | 404 | 6,051 (4,096) | ? | 1,364 | **4.4x** | 15x |
| FetchPushDict | Fetch | 480x480 | ? | 370 | 6,053 (4,096) | ? | 1,365 | **4.4x** | 16x |
| FetchPush | Fetch | 480x480 | ? | 330 | 6,053 (1,024) | ? | 1,390 | **4.4x** | 18x |
| FetchPickAndPlaceDict | Fetch | 480x480 | ? | 358 | 6,008 (4,096) | ? | 1,392 | **4.3x** | 17x |
| FetchPickAndPlace | Fetch | 480x480 | ? | 327 | 6,008 (4,096) | ? | 1,399 | **4.3x** | 18x |
| MujocoHalfCheetah | MuJoCo locomotion | 240x240 | ? | 379 | 10,931 (4,096) | ? | 2,581 | **4.2x** | 29x |
| DMCCartpoleBalance | DM Control | 84x84 | 2 | 353 | 28,883 (4,096) | 100% | 7,249 | **4.0x** | 82x |
| HumanoidBenchH1Stand | HumanoidBench | 256x256 | ? | 31 | 16,726 (256) | ? | 4,301 | **3.9x** | 547x |
| Parking | highway-env | 300x600 | ? | — | 7,398 (4,096) | ? | 1,914 | **3.9x** | — |
| HumanoidBenchH1SitSimple | HumanoidBench | 256x256 | ? | 38 | 12,023 (4,096) | ? | 3,125 | **3.8x** | 312x |
| MetaWorldPushWall | Meta-World | 480x480 | ? | 83 | 2,512 (4,096) | ? | 655 | **3.8x** | 30x |
| MetaWorldReachWall | Meta-World | 480x480 | ? | 8 | 2,508 (4,096) | ? | 664 | **3.8x** | 325x |
| HumanoidBenchH1Stair | HumanoidBench | 256x256 | ? | 58 | 11,857 (1,024) | ? | 3,202 | **3.7x** | 203x |
| Racetrack | highway-env | 600x600 | ? | — | 2,582 (4,096) | ? | 708 | **3.6x** | — |
| MetaWorldPickPlace | Meta-World | 480x480 | ? | 68 | 2,525 (4,096) | ? | 711 | **3.5x** | 37x |
| MujocoWalker2d | MuJoCo locomotion | 240x240 | ? | 373 | 10,887 (4,096) | ? | 3,067 | **3.5x** | 29x |
| MujocoSwimmer | MuJoCo locomotion | 240x240 | ? | 356 | 10,884 (4,096) | ? | 3,177 | **3.4x** | 31x |
| MetaWorldPickOutOfHole | Meta-World | 480x480 | ? | 122 | 2,447 (4,096) | ? | 721 | **3.4x** | 20x |
| DMCPointMassEasy | DM Control | 84x84 | 2 | 370 | 25,768 (4,096) | 100% | 7,598 | **3.4x** | 70x |
| HumanoidBenchH1HandWalk | HumanoidBench | 256x256 | ? | 36 | 14,412 (256) | ? | 4,258 | **3.4x** | 395x |
| HumanoidBenchH1HandRun | HumanoidBench | 256x256 | ? | 35 | 14,422 (256) | ? | 4,311 | **3.3x** | 414x |
| MujocoHumanoid | MuJoCo locomotion | 240x240 | ? | 329 | 9,645 (4,096) | ? | 2,906 | **3.3x** | 29x |
| MujocoAnt | MuJoCo locomotion | 240x240 | ? | 340 | 9,936 (4,096) | ? | 3,011 | **3.3x** | 29x |
| MetaWorldDoorLock | Meta-World | 480x480 | ? | 24 | 2,279 (4,096) | ? | 703 | **3.2x** | 96x |
| MetaWorldHandlePress | Meta-World | 480x480 | ? | 21 | 2,329 (4,096) | ? | 728 | **3.2x** | 112x |
| MetaWorldBinPicking | Meta-World | 480x480 | ? | 49 | 2,489 (1,024) | ? | 785 | **3.2x** | 51x |
| HumanoidBenchH1HandSpoon | HumanoidBench | 256x256 | ? | 37 | 9,330 (256) | ? | 2,947 | **3.2x** | 255x |
| MetaWorldCoffeeButton | Meta-World | 480x480 | ? | 29 | 2,369 (4,096) | ? | 750 | **3.2x** | 82x |
| RoboCasaCounterToSinkWrist | RoboCasa | 256x256 | ? | 560 | 11,687 (1,024) | ? | 3,718 | **3.1x** | 21x |
| MetaWorldButtonPressTopdown | Meta-World | 480x480 | ? | 153 | 2,417 (4,096) | ? | 770 | **3.1x** | 16x |
| MetaWorldDrawerOpen | Meta-World | 480x480 | ? | 24 | 2,427 (4,096) | ? | 774 | **3.1x** | 103x |
| HumanoidBenchH1HandBalanceSimple | HumanoidBench | 256x256 | ? | 35 | 9,887 (256) | ? | 3,172 | **3.1x** | 283x |
| MetaWorldSweep | Meta-World | 480x480 | ? | 21 | 2,534 (1,024) | ? | 829 | **3.1x** | 121x |
| MetaWorldStickPush | Meta-World | 480x480 | ? | 212 | 2,198 (4,096) | ? | 723 | **3.0x** | 10x |
| MetaWorldFaucetClose | Meta-World | 480x480 | ? | 57 | 2,423 (4,096) | ? | 797 | **3.0x** | 42x |
| MetaWorldPickPlaceWall | Meta-World | 480x480 | ? | 64 | 2,512 (4,096) | ? | 832 | **3.0x** | 39x |
| MetaWorldButtonPressTopdownWall | Meta-World | 480x480 | ? | 27 | 2,404 (4,096) | ? | 805 | **3.0x** | 90x |
| HumanoidBenchH1SitHard | HumanoidBench | 256x256 | ? | 38 | 12,060 (1,024) | ? | 4,056 | **3.0x** | 314x |
| MujocoReacher | MuJoCo locomotion | 240x240 | 2 | 2,074 | 7,756 (16,384) | 100% | 2,751 | **2.8x** | 4x |
| MetaWorldHandlePullSide | Meta-World | 480x480 | ? | 219 | 2,331 (4,096) | ? | 831 | **2.8x** | 11x |
| MetaWorldPlateSlideSide | Meta-World | 480x480 | ? | 216 | 2,228 (4,096) | ? | 801 | **2.8x** | 10x |
| MujocoHopper | MuJoCo locomotion | 240x240 | ? | 356 | 10,566 (4,096) | ? | 3,846 | **2.7x** | 30x |
| MetaWorldPlateSlideBack | Meta-World | 480x480 | ? | 217 | 2,207 (4,096) | ? | 805 | **2.7x** | 10x |
| MetaWorldButtonPress | Meta-World | 480x480 | ? | 23 | 2,388 (4,096) | ? | 872 | **2.7x** | 102x |
| MetaWorldSweepInto | Meta-World | 480x480 | ? | 29 | 2,554 (1,024) | ? | 950 | **2.7x** | 88x |
| DMCAcrobotSwingup | DM Control | 84x84 | 2 | 348 | 19,226 (4,096) | 100% | 7,235 | **2.7x** | 55x |
| MetaWorldPush | Meta-World | 480x480 | ? | 135 | 2,522 (4,096) | ? | 957 | **2.6x** | 19x |
| MetaWorldReach | Meta-World | 480x480 | ? | 216 | 2,515 (4,096) | ? | 957 | **2.6x** | 12x |
| HumanoidBenchH1BalanceSimple | HumanoidBench | 256x256 | ? | 37 | 11,408 (4,096) | ? | 4,342 | **2.6x** | 307x |
| MetaWorldHandlePull | Meta-World | 480x480 | ? | 152 | 2,338 (4,096) | ? | 898 | **2.6x** | 15x |
| MetaWorldDoorClose | Meta-World | 480x480 | ? | 21 | 2,260 (4,096) | ? | 868 | **2.6x** | 108x |
| MetaWorldPushBack | Meta-World | 480x480 | ? | 84 | 2,525 (4,096) | ? | 974 | **2.6x** | 30x |
| MetaWorldShelfPlace | Meta-World | 480x480 | ? | 38 | 2,369 (4,096) | ? | 915 | **2.6x** | 62x |
| MetaWorldAssembly | Meta-World | 480x480 | ? | 23 | 2,256 (4,096) | ? | 871 | **2.6x** | 98x |
| MetaWorldPegInsertSide | Meta-World | 480x480 | ? | 26 | 2,435 (4,096) | ? | 944 | **2.6x** | 95x |
| HumanoidBenchH1HandSitSimple | HumanoidBench | 256x256 | ? | 38 | 10,551 (64) | ? | 4,107 | **2.6x** | 280x |
| MetaWorldDoorUnlock | Meta-World | 480x480 | ? | 26 | 2,278 (4,096) | ? | 887 | **2.6x** | 87x |
| Intersection | highway-env | 600x600 | ? | — | 2,249 (4,096) | ? | 881 | **2.6x** | — |
| MetaWorldPegUnplugSide | Meta-World | 480x480 | ? | 26 | 2,397 (4,096) | ? | 949 | **2.5x** | 91x |
| MetaWorldLeverPull | Meta-World | 480x480 | ? | 63 | 2,354 (4,096) | ? | 947 | **2.5x** | 37x |
| MetaWorldHandInsert | Meta-World | 480x480 | ? | 23 | 2,542 (1,024) | ? | 1,030 | **2.5x** | 111x |
| MetaWorldPlateSlideBackSide | Meta-World | 480x480 | ? | 215 | 2,236 (4,096) | ? | 916 | **2.4x** | 10x |
| MetaWorldPlateSlide | Meta-World | 480x480 | ? | 214 | 2,221 (4,096) | ? | 933 | **2.4x** | 10x |
| MetaWorldBasketball | Meta-World | 480x480 | ? | 34 | 2,423 (4,096) | ? | 1,020 | **2.4x** | 72x |
| MetaWorldFaucetOpen | Meta-World | 480x480 | ? | 27 | 2,428 (4,096) | ? | 1,023 | **2.4x** | 90x |
| MetaWorldDialTurn | Meta-World | 480x480 | ? | 21 | 2,503 (4,096) | ? | 1,066 | **2.3x** | 117x |
| MetaWorldCoffeePull | Meta-World | 480x480 | ? | 27 | 2,368 (4,096) | ? | 1,010 | **2.3x** | 86x |
| MetaWorldButtonPressWall | Meta-World | 480x480 | ? | 22 | 2,369 (4,096) | ? | 1,013 | **2.3x** | 107x |
| HumanoidBenchH1HandStair | HumanoidBench | 256x256 | ? | 57 | 10,347 (4,096) | ? | 4,424 | **2.3x** | 183x |
| MetaWorldCoffeePush | Meta-World | 480x480 | ? | 22 | 2,362 (4,096) | ? | 1,012 | **2.3x** | 107x |
| MetaWorldBoxClose | Meta-World | 480x480 | ? | 22 | 2,437 (4,096) | ? | 1,050 | **2.3x** | 113x |
| Roundabout | highway-env | 600x600 | ? | — | 1,520 (4,096) | ? | 660 | **2.3x** | — |
| DMCCartpoleSwingup | DM Control | 84x84 | 2 | 343 | 23,354 (4,096) | 100% | 10,140 | **2.3x** | 68x |
| MetaWorldStickPull | Meta-World | 480x480 | ? | 20 | 2,202 (4,096) | ? | 957 | **2.3x** | 112x |
| MetaWorldDrawerClose | Meta-World | 480x480 | ? | 25 | 2,415 (4,096) | ? | 1,052 | **2.3x** | 96x |
| MetaWorldDoorOpen | Meta-World | 480x480 | ? | 26 | 2,304 (4,096) | ? | 1,012 | **2.3x** | 89x |
| MetaWorldSoccer | Meta-World | 480x480 | ? | 216 | 2,101 (4,096) | ? | 930 | **2.3x** | 10x |
| MujocoInvertedPendulum | MuJoCo locomotion | 240x240 | 2 | 2,591 | 7,652 (16,384) | 100% | 3,389 | **2.3x** | 3x |
| MetaWorldHandlePressSide | Meta-World | 480x480 | ? | 23 | 2,340 (4,096) | ? | 1,038 | **2.3x** | 104x |
| MetaWorldDisassemble | Meta-World | 480x480 | ? | 123 | 2,229 (4,096) | ? | 1,031 | **2.2x** | 18x |
| MetaWorldHammer | Meta-World | 480x480 | ? | 21 | 2,183 (1,024) | ? | 1,014 | **2.2x** | 102x |
| DMCPendulumSwingup | DM Control | 84x84 | 2 | 349 | 22,336 (4,096) | 100% | 10,430 | **2.1x** | 64x |
| RoboCasaCounterToSinkRight | RoboCasa | 256x256 | ? | 455 | 5,122 (1,024) | ? | 2,511 | **2.0x** | 11x |
| DMCFingerTurnHard | DM Control | 84x84 | 2 | 349 | 13,442 (4,096) | 100% | 7,067 | **1.9x** | 38x |
| DMCReacherHard | DM Control | 84x84 | 2 | 355 | 16,159 (4,096) | 100% | 8,691 | **1.9x** | 46x |
| MujocoPusher | MuJoCo locomotion | 240x240 | 2 | 468 | 4,404 (16,384) | 100% | 2,390 | **1.8x** | 9x |
| RoboCasaCounterToSinkL45S33 | RoboCasa | 256x256 | ? | 403 | 5,393 (4,096) | ? | 2,930 | **1.8x** | 13x |
| RoboCasaCounterToSink | RoboCasa | 256x256 | ? | 87 | 5,193 (1,024) | ? | 3,038 | **1.7x** | 60x |
| DMCFingerTurnEasy | DM Control | 84x84 | 2 | 353 | 13,368 (4,096) | 100% | 8,282 | **1.6x** | 38x |
| DMCFingerSpin | DM Control | 84x84 | 2 | 358 | 15,359 (4,096) | 100% | 9,854 | **1.6x** | 43x |
| MujocoInvertedDoublePendulum | MuJoCo locomotion | 240x240 | 2 | 2,014 | 7,527 (4,096) | 100% | 4,834 | **1.6x** | 4x |
| MetaWorldWindowOpen | Meta-World | 480x480 | ? | 90 | 1,039 (4,096) | ? | 697 | **1.5x** | 12x |
| DMCReacherEasy | DM Control | 84x84 | 2 | 360 | 15,355 (256) | 100% | 10,396 | **1.5x** | 43x |
| MetaWorldWindowClose | Meta-World | 480x480 | ? | 200 | 1,050 (4,096) | ? | 909 | **1.2x** | 5x |
| DMCBallInCupCatch | DM Control | 84x84 | 2 | 345 | 3,813 (4,096) | 150% | 10,174 | **0.4x** | 11x |
| CarRacing | CarRacing | 96x96 | ? | 133 | 179 (1,024) | ? | 9,884 | **0.0x** | 1x |

## By family

| family | envs | frame | default f/s | ours f/s | stored f/s/core | **cores replaced** |
|---|---|---|---|---|---|---|
| Griddly | 7 | 144x120… | 99–145 | 2,307,114–9,438,816 | 851–5,276 | **1,789.0–3,647.9x** |
| Sokoban | 1 | 160x160 | 253 | 9,571,978 | 3,983 | **2,403.2x** |
| Overcooked | 5 | 192x224… | 41–1,595 | 3,615,605–8,145,368 | 2,914–4,399 | **1,241.0–1,860.7x** |
| MiniGrid / BabyAI | 7 | 224x224… | 117–942 | 711,991–4,479,313 | 608–4,011 | **329.7–1,225.8x** |
| MiniHack | 1 | 336x1264 | — | 244,456 | 698 | **350.0x** |
| Jumanji | 8 | 200x200 | 3–88 | 409,277–1,286,366 | 3,086–4,387 | **105.2–340.4x** |
| Procgen | 4 | 64x64 | 899–975 | 313,372–1,588,402 | 8,344–11,055 | **28.3–190.4x** |
| PushT | 1 | 96x96 | 472 | 1,256,557 | 10,259 | **122.5x** |
| OGBench | 25 | 224x224 | 82–491 | 48,978–206,740 | 1,893–4,249 | **11.9–93.6x** |
| swm nav | 5 | 224x224… | 52–772 | 71,140–187,629 | 1,083–4,985 | **15.3–86.7x** |
| Crafter | 1 | 64x64 | — | 665,836 | 8,186 | **81.3x** |
| Craftax | 1 | 112x144 | — | 569,934 | 7,039 | **81.0x** |
| PGX | 4 | 240x240… | 21–108 | 78,322–109,484 | 2,055–3,160 | **25.0–45.7x** |
| classic control | 5 | 400x600… | 249–541 | 33,216–39,843 | 1,120–1,626 | **22.3–35.6x** |
| D4RL PointMaze | 4 | 224x224 | 1,846–2,238 | 88,303–88,935 | 2,919–4,403 | **20.1–30.3x** |
| DM Control | 16 | 84x84 | 272–370 | 3,813–140,391 | 6,110–10,430 | **0.4–23.0x** |
| LunarLander | 1 | 400x600 | 336 | 18,786 | 1,165 | **16.1x** |
| highway-env | 6 | 150x600… | — | 1,520–20,389 | 660–3,205 | **2.3–7.2x** |
| Fetch | 12 | 480x480 | 323–404 | 6,008–6,164 | 896–1,399 | **4.3–6.9x** |
| HumanoidBench | 17 | 256x256 | 31–59 | 9,330–16,835 | 2,347–4,424 | **2.3–6.8x** |
| MuJoCo locomotion | 10 | 240x240 | 329–2,591 | 4,404–10,931 | 2,390–4,834 | **1.6–4.2x** |
| Meta-World | 50 | 480x480 | 8–219 | 1,039–2,554 | 655–1,066 | **1.2–3.8x** |
| RoboCasa | 4 | 256x256 | 87–560 | 5,122–11,687 | 2,511–3,718 | **1.7–3.1x** |
| CarRacing | 1 | 96x96 | 133 | 179 | 9,884 | **0.0x** |

## Does the *load* half of load-and-decode matter?

Across 196 envs, reading a warm local JPEG file and decoding it runs at 0.45–1.95x the speed of decoding the same bytes already in memory (median 0.96x).
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
