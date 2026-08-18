# `stable_envs.render.gfxdraw`

An exact batched reimplementation of the SDL_gfx primitives `pygame.gfxdraw` exposes, for envs whose
reference renders through it — gymnasium's classic-control family today.

It sits beside `raster2d` rather than inside it because the claim is different in kind. `gfxdraw` takes
`Sint16` vertices, so every coordinate is truncated to an integer before SDL_gfx sees it and everything
downstream is integer arithmetic: there is no sub-pixel coverage to approximate and therefore no
tolerance to hide behind. Redoing the same integer arithmetic on the GPU is the same computation, and the
result is 0 differing pixels over 250.9 M — not a high percentage.

::: stable_envs.render.gfxdraw
