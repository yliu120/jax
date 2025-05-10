from __future__ import annotations

from collections.abc import Callable, Sequence
import functools
from functools import partial
import logging
from typing import Any
import types

import numpy as np

from jax._src import ad_util
from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src import effects
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import api_util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax as lax_internal
from jax._src.lax import convolution as lax_convolution
from jax._src.lib.mlir.dialects import hlo
from jax._src.traceback_util import api_boundary
from jax._src.tree_util import PyTreeDef, tree_flatten, tree_unflatten, tree_structure
from jax._src.util import (unzip2, wraps, split_list, partition_list, safe_map,
                           safe_zip, merge_lists, weakref_lru_cache)

source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)

map = safe_map
zip = safe_zip

logger = logging.getLogger(__name__)


pipeline_p = core.Primitive('pipeline')
pipeline_p.multiple_results = True


@pipeline_p.def_impl
def pipeline_impl(*args, jaxpr, differentiated):
  del differentiated  # Unused.
  return core.eval_jaxpr(jaxpr, (), *args)


@pipeline_p.def_effectful_abstract_eval
def pipeline_abstrace_eval(*args, jaxpr, differentiated):
  del args, differentiated, # Unused.
  return [v.aval for v in jaxpr.outvars], jaxpr.effects


def pipeline_jvp(primals, tangents, jaxpr, differentiated):
  assert not jaxpr.constvars
  in_nonzeros = [type(t) is not ad_util.Zero for t in tangents]
  jaxpr_jvp_, out_nz = ad.jvp_jaxpr(pe.close_jaxpr(jaxpr), in_nonzeros, False)
  nonzero_tangents = [t for t in tangents if type(t) is not ad_util.Zero]
  jaxpr_jvp = pe.convert_constvars_jaxpr(jaxpr_jvp_.jaxpr)
  outs = pipeline_p.bind(
      *jaxpr_jvp_.consts, *primals, *nonzero_tangents, jaxpr=jaxpr_jvp,
      differentiated=differentiated)
  out_primals, out_tangents_ = split_list(outs, [len(jaxpr.outvars)])
  out_tangents_ = iter(out_tangents_)
  out_tangents = [next(out_tangents_) if nz else ad_util.Zero.from_primal_value(p)
                  for p, nz in zip(out_primals, out_nz)]
  return out_primals, out_tangents
ad.primitive_jvps[pipeline_p] = pipeline_jvp


def pipeline_partial_eval(trace: pe.JaxprTrace, *tracers: core.Tracer,
                       jaxpr: core.Jaxpr, **params):
  assert not jaxpr.constvars
  in_unknowns = [not t.is_known() for t in tracers]
  jaxpr_known, jaxpr_staged, out_unknowns, out_inst, num_res = \
      pe.partial_eval_jaxpr_custom(
          jaxpr, in_unknowns, [True] * len(in_unknowns), False, False, policy)

  # DCE jaxpr_staged, keeping only instantiated outputs which are unknown
  _, out_inst_unknown = partition_list(out_inst, out_unknowns)
  jaxpr_unknown, in_used_staged = pe.dce_jaxpr(jaxpr_staged, out_inst_unknown)
  used_res, in_used_staged = split_list(in_used_staged, [num_res])

  # DCE jaxpr_known, keeping all known outputs but discarding dce'd res
  out_used_known = [True] * (len(out_unknowns) - sum(out_unknowns)) + used_res
  jaxpr_known, in_used_known = pe.dce_jaxpr(jaxpr_known, out_used_known)
  num_res = sum(used_res)

  # compute known outputs and residuals (hoisted out of remat primitive)
  _, in_consts_ = unzip2(t.pval for t in tracers if t.pval.is_known())
  _, in_consts = partition_list(in_used_known, in_consts_)
  out_consts = core.eval_jaxpr(jaxpr_known, (), *in_consts)
  out_knowns, residuals = split_list(out_consts, [len(out_consts)-num_res])

  # set up unknown outputs with a recipe to call remat
  res_tracers = map(trace.new_instantiated_const, residuals)
  _, tracers_staged = partition_list(in_used_staged, tracers)
  in_jaxpr_tracers = res_tracers + map(trace.instantiate_const, tracers_staged)  # type: ignore
  out_jaxpr_tracers = [pe.JaxprTracer(trace, pe.PartialVal.unknown(x.aval), None)
                       for x in jaxpr_unknown.outvars]
  new_params = dict(params, jaxpr=jaxpr_unknown, differentiated=True)
  recipe = pe.new_eqn_recipe(in_jaxpr_tracers, out_jaxpr_tracers, pipeline_p,
                             new_params, jaxpr_unknown.effects,
                             source_info_util.current())

  for t in out_jaxpr_tracers: t.recipe = recipe

  # zip together known and unknown outputs
  return merge_lists(out_unknowns, out_knowns, out_jaxpr_tracers)
pe.custom_partial_eval_rules[pipeline_p] = pipeline_partial_eval


def _pipeline_lowering(
    ctx,
    *args,
    jaxpr: core.Jaxpr,
    differentiated: bool,
):
  jaxpr_args: Sequence[mlir.IrValues]
  if differentiated:
    arg_types = map(mlir.aval_to_ir_type, ctx.avals_in)
    flat_args = mlir.flatten_ir_values(args)
    jaxpr_args = mlir.unflatten_ir_values_like_types(
      flat_args, arg_types)
  else:
    jaxpr_args = args
  outs, tokens_out = mlir.jaxpr_subcomp(
      ctx.module_context, jaxpr, ctx.name_stack.extend('pipelined'),
      ctx.tokens_in, (), *jaxpr_args, dim_var_values=ctx.dim_var_values)
  ctx.set_tokens_out(tokens_out)
  return outs


mlir.register_lowering(pipeline_p, _pipeline_lowering)


@weakref_lru_cache
def _trace_to_jaxpr(fun: Callable,
                    in_tree: PyTreeDef,
                    in_avals: Sequence[core.AbstractValue],
                    debug: core.DebugInfo
                    ) -> tuple[core.Jaxpr, Sequence[Any], PyTreeDef]:
  flat_fun, out_tree = api_util.flatten_fun(lu.wrap_init(fun, debug_info=debug), in_tree)
  jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
  return pe.convert_constvars_jaxpr(jaxpr), consts, out_tree()


@api_boundary
def pipeline(fun: Callable) -> Callable:
  @wraps(fun)
  @api_boundary
  def fun_pipeline(*args, **kwargs):
    debug = api_util.debug_info(
        "pipelined", fun, args, kwargs)
    args_flat, in_tree = tree_flatten((args, kwargs))
    in_avals = [core.shaped_abstractify(x) for x in args_flat]
    jaxpr, consts, out_tree = _trace_to_jaxpr(fun, in_tree, tuple(in_avals), debug)
    out_flat = pipeline_p.bind(*consts, *args_flat, jaxpr=jaxpr,differentiated=False)
    return tree_unflatten(out_tree, out_flat)
  return fun_pipeline
