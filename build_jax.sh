#!/bin/bash

export XLA_DIR=/data/yunlongl/workspace/xla

# Let's keep using monolithic jaxlib until jax plugins ready.
rm -rf dist

python build/build.py build \
  --wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt \
  --use_clang \
  --bazel_options=--override_repository=xla=${XLA_DIR} \
  --bazel_options=--repo_env=LOCAL_CUDA_PATH="/usr/local/cuda" \
  --bazel_options=--repo_env=LOCAL_CUDNN_PATH="/usr/local/cuda/cudnn" \
  --bazel_options=--repo_env=LOCAL_NCCL_PATH="/usr/local"

pip install --no-deps --force-reinstall dist/*.whl
pip install -e . --no-deps --force-reinstall
