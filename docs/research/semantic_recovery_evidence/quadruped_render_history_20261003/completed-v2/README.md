# Quadruped render-history diagnostic reproduced

Job 6967633 on the original RTX 3090 host gpu2109 exits FAILED 1:0 because the
unchanged frozen gate still raises its exact-frame assertion. The diagnostic
itself completes its trace; this is not acceptance and no default is enabled.

216 renders form 32 identical-input groups. The first occurrence of one pose
(record 30) differs from five later occurrences (62/94/126/158/190) in two
channels. Single-frame normalized MSE is 2.503043701140587e-9; the earlier
32-frame aggregate was 7.822011566064334e-11. These have different denominators.

For each mismatching pair, compact State/style/pristine inputs have identical
hashes, captured scene fields before and after rendering are identical, and
captured kinematics are identical. The repeated images subsequently agree.
This narrows the investigation toward render-context/history or uninstrumented
GL state rather than exported simulator values. It does not identify the driver
mechanism or prove all hidden renderer state equal. Keep the exact assertion;
the next diagnostic should examine context initialization and GL state/history.
The source files, arrays, original failure and trace remain intact.
