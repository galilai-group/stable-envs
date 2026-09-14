# Round 3, GPU half — superseded

Job `6352385`, `gpu2107`. The code is the same as the authoritative pair (`c96f886` against
`1103326`), and the CPU half of this job IS authoritative and sits in the parent directory. Only the
GPU half is superseded, and for a methodological reason rather than a code one: it ran `before` then
`after` once each, so the `after` leg always had the warmer card. Every row came back 0.96-1.00,
including the two whose cuda code is byte-identical between the trees, which is how the bias was
spotted. `gpu_after_first.json` / `gpu_before_second.json` in the parent directory repeat the pair
with the order reversed.
