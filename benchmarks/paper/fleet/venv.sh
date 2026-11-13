#!/bin/bash
# Build one family venv for the fleet with the recipe of venv() in benchmarks/paper/run.sh, into $VENVS/venv-FAMILY.
# usage: venv.sh FAMILY    (env: SNV_REPO, SWM, UV, VENVS, TORCH_VERSION; NVDIFFRAST_SITE: a prebuilt nvdiffrast, else
# it is built here)
# run.sh then uses it through VENV_PYTHON. Keep this recipe the same as run.sh's.
set -uo pipefail
fam=$1; REPO=$SNV_REPO; v=$VENVS/venv-$fam; log=$VENVS/venv-$fam.log
case $fam in
  classic) extras="classic"; groups="pin-classic";;
  navigation) extras="navigation navigation-gpu"; groups="pin-navigation";;
  pusht) extras="pusht pusht-gpu"; groups="pin-pusht";;
  dmc) extras="dmcontrol dmcontrol-gpu"; groups="pin-dmc gpu-mjwarp";;
  ogbench) extras="ogbench ogbench-manipulation ogbench-manipulation-gpu"; groups="pin-ogbench";;
  fetch) extras="robotics mjwarp"; groups="pin-fetch";;
  rocket) extras="rocket"; groups="pin-rocket";;
  *) echo "unknown family $fam" >&2; exit 1;;
esac
[ -s $v.ok ] && { echo $v/bin/python; exit 0; }
rm -rf $v
# shellcheck disable=SC2046
( cd "$REPO" && UV_PROJECT_ENVIRONMENT=$v $UV sync --frozen --python 3.12 --no-default-groups \
    $(for e in $extras; do echo --extra $e; done) $(for g in $groups torch-cuda test; do echo --group $g; done) ) \
    > $log 2>&1 || { echo "venv $fam failed" >&2; exit 1; }
tv=${TORCH_VERSION:-2.14.1}
cu=(--extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match)
$UV pip install --python $v/bin/python "${cu[@]}" --reinstall-package torch "torch==$tv+cu126" >> $log 2>&1 \
  || { echo "torch $fam failed" >&2; exit 1; }
$UV pip freeze --python $v/bin/python | grep "==" > $VENVS/constraints-$fam.txt
tvv=$(echo "$tv" | awk -F. '{print "0." $2 + 15 "." $3}')
$UV pip install --python $v/bin/python "${cu[@]}" -c $VENVS/constraints-$fam.txt -e "$SWM" "pymunk==7.2.0" shapely \
  opencv-python-headless "torchvision==$tvv+cu126" >> $log 2>&1 || { echo "swm $fam failed" >&2; exit 1; }
if [ $fam = rocket ]; then
  sp=$($v/bin/python -c "import site; print(site.getsitepackages()[0])")
  if [ -d "${NVDIFFRAST_SITE:-}/nvdiffrast" ]; then
    cp -r $NVDIFFRAST_SITE/{nvdiffrast,nvdiffrast-0.4.0.dist-info,_nvdiffrast_c*.so} "$sp"/ 2>/dev/null
  else
    # Build nvdiffrast for the GPUs of this machine; it needs nvcc (the CUDA toolkit).
    $UV pip install --python $v/bin/python setuptools wheel ninja >> $log 2>&1
    arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | tr '\n' ';' | sed 's/;$//')
    PATH=$PATH:/usr/local/cuda/bin TORCH_CUDA_ARCH_LIST="$arch" $UV pip install --python $v/bin/python --no-build-isolation \
      "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae" >> $log 2>&1
  fi
  $UV pip install --python $v/bin/python "${cu[@]}" -c $VENVS/constraints-$fam.txt cvxpy >> $log 2>&1
  # Without nvdiffrast only the Rocket GPU frames cells fail; the other cells still run.
  $v/bin/python -c "import nvdiffrast.torch" >> $log 2>&1 || echo "nvdiffrast missing: Rocket GPU frames will fail" >&2
fi
$v/bin/python -c "import torch, stable_envs, stable_worldmodel" >> $log 2>&1 || { echo "imports $fam failed" >&2; exit 1; }
$UV pip freeze --python $v/bin/python > $VENVS/freeze-$fam.txt
date -Is > $v.ok
echo $v/bin/python
