# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for distributed multi-process jit cache."""

import functools
import os

import jax
from jax._src import test_multiprocess as jt_multiprocess
import jax.numpy as jnp


P = jax.sharding.PartitionSpec
partial = functools.partial


class JitCacheTest(jt_multiprocess.MultiProcessTest):

  def test_jit_cache(self):
    jax.config.update("jax_compilation_cache_dir", "/tmp/compilation_cache")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    mesh = jax.make_mesh((4,), ('x'))
    in_out_sharding = jax.sharding.NamedSharding(mesh, P('x'))
    f = jax.jit(lambda x: x, in_shardings=(in_out_sharding,), out_shardings=in_out_sharding)
    jax.block_until_ready(f(jnp.arange(4)))


if __name__ == '__main__':
  # This test doesn't work with the platform allocator, so we override it
  # if it's ran alone. If it's part of a larger test suite and the platform
  # allocator is used, setUp will skip the test.
  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'
  os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'default'
  jt_multiprocess.main()
