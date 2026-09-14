#!/bin/bash
# Regenerates the deliverable tables from the committed artifacts. Run from the worktree root.
set -eu
R=examples/results/step6/lance
python3 scratch/step6/lance/compare2.py --root . \
  --lance-cpu        $R/lance_cpu_node2317_6366909.json \
  --lance-gpu        $R/lance_gpu_gpu2109_6366910.json \
  --lance-arms2-cpu  $R/lance_arms2_node1956_6367705.json \
  --lance-arms2-gpu  $R/lance_arms2_gpu2104_6367697.json \
  --cpu-8way         $R/renderer_cpu_8way_node1956_6367705.json \
  --cpu1-shard       $R/renderer_cpu_1thread_r2_node1956_6367705.json \
  --gpu-shard        $R/renderer_gpu_after_gpu2104_6366975.json \
  --gpu-shard        $R/renderer_gpu_after_fetch311_gpu2104_6366975.json \
  --pipeline         $R/pipeline_gpu_r4_"$1".json \
  --pipeline         $R/pipeline_gpu_r4_fetch311_"$1".json \
  --jpeg-to-gpu      $R/jpeg_to_gpu_"$1".json \
  --quality-probe    $R/quality_probe_node2349_6366960.json
