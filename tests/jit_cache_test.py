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
import jax.numpy as jnp
from jax._src import test_multiprocess as jt_multiprocess
from jax.experimental.serialize_executable import (
    deserialize_and_load,
    serialize,
)
from jax.experimental.multihost_utils import host_local_array_to_global_array

import numpy as np


P = jax.sharding.PartitionSpec
partial = functools.partial


class JitCacheTest(jt_multiprocess.MultiProcessTest):

  def test_jit_cache(self):
    jax.config.update("jax_compilation_cache_dir", "/tmp/compilation_cache")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    mesh = jax.make_mesh((4,), ('x'))
    in_out_sharding = jax.sharding.NamedSharding(mesh, P('x'))
    f = jax.jit(lambda x: x, in_shardings=(in_out_sharding,), out_shardings=in_out_sharding)
    inp = host_local_array_to_global_array(
        np.arange(1) * jax.process_index(), mesh, in_out_sharding.spec)
    jax.block_until_ready(f(inp))

  def test_jit_cache_inline(self):
    jax.config.update("jax_compilation_cache_dir", "/tmp/compilation_cache")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    mesh = jax.make_mesh((4,), ('x'))
    in_out_sharding = jax.sharding.NamedSharding(mesh, P('x'))
    f = jax.jit(lambda x: x, in_shardings=(in_out_sharding,), out_shardings=in_out_sharding)
    input_data_gen = jax.jit(lambda: jnp.arange(4), out_shardings=in_out_sharding)
    jax.block_until_ready(f(input_data_gen()))

  def test_aot_route(self):
    def fun(x):
      return x * x

    mesh = jax.make_mesh((4,), ('x'))
    in_out_sharding = jax.sharding.NamedSharding(mesh, P('x', None))
    lowered = jax.jit(
      fun, in_shardings=in_out_sharding, out_shardings=in_out_sharding
    ).lower(jax.core.ShapedArray(shape=(8, 8), dtype=jnp.float32))

    def verify_serialization(lowered):
      serialized, in_tree, out_tree = serialize(lowered.compile())
      compiled = deserialize_and_load(serialized, in_tree, out_tree)
      self.assertEqual(compiled.as_text(), lowered.compile().as_text())
      return compiled
    
    compiled = verify_serialization(lowered)
    global_inp_data = np.arange(64, dtype=jnp.float32).reshape(8, 8)

    def cb(index):
      return global_inp_data[index]
    inp = jax.make_array_from_callback((8, 8), in_out_sharding, cb)
    jax.block_until_ready(compiled(inp))


if __name__ == '__main__':
  # This test doesn't work with the platform allocator, so we override it
  # if it's ran alone. If it's part of a larger test suite and the platform
  # allocator is used, setUp will skip the test.
  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'
  os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'default'
  jt_multiprocess.main()
