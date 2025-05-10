from __future__ import annotations

from functools import partial

from absl import logging
from absl.testing import absltest, parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import api, api_util
from jax._src import test_util as jtu
from jax._src.pipeline import pipeline

from jax.ad_checkpoint import checkpoint_name, checkpoint as new_checkpoint

import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

config.parse_flags_with_absl()


class PipelineTest(jtu.JaxTestCase):

  def test_pipeline_scan_for_residuals(self):
    to_scan = lambda c, _: (jnp.sin(c), None)

    def f_noremat(x):
      y, _ = lax.scan(to_scan, x, np.arange(3.))
      return y

    pipelined_f = pipeline(f_noremat)
    print(api.make_jaxpr(api.value_and_grad(pipelined_f))(4.))
