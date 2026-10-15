# `stable_envs.render.gfxdraw`

This module is a batched reimplementation of the SDL_gfx primitives that `pygame.gfxdraw` exposes. Gymnasium's
classic-control family renders through it.

It is separate from `raster2d` because the arithmetic is integer. `gfxdraw` takes `Sint16` vertices,
so the module truncates every coordinate before drawing. The GPU path repeats the same integer arithmetic, so
the output is exact. The conformance comparison found 0 differing pixels over 250.9 M.

::: stable_envs.render.gfxdraw
