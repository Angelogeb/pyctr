import ast
import inspect
import sys
from textwrap import dedent

import torch
from torch._C._jit_tree_views import (
    Assign,
    BinOp,
    Decl,
    Def,
    For,
    Ident,
    If,
    Return,
    TupleLiteral,
    Var,
)
from torch.jit.frontend import SourceContext

from pyctr.api import conversion
from pyctr.examples.torchscript import call_helper
from pyctr.examples.torchscript.dmmy import ctx, dmmy_rng
from pyctr.examples.torchscript.expression import TorchExpr, torch_expr
from pyctr.examples.torchscript.torch_ast import emit_node, flush_ast, fresh_ast
from pyctr.overloads import py_defaults
from pyctr.transformers.virtualization import control_flow, functions, variables

tch_ir_ = sys.modules[__name__]


class Rep(TorchExpr):
    def __init__(self, val):
        super().__init__(val)


init = py_defaults.init


def assign(lhs, rhs):
    if isinstance(rhs, Rep):
        rhs = TorchExpr(Var(Ident(dmmy_rng, lhs.name)))
    elif isinstance(rhs, TorchExpr):
        assignment = Assign([Var(Ident(dmmy_rng, lhs.name))], rhs.node)
        emit_node(assignment)
        rhs = TorchExpr(Var(Ident(dmmy_rng, lhs.name)))

    return py_defaults.assign(lhs, rhs)


call = call_helper.call


def read(var):
    if isinstance(var.val, torch.Tensor) or for_var_marker.is_marked(var):
        return TorchExpr(Var(Ident(dmmy_rng, var.name)))
    return var.val


class Marker:
    pass


class ForVarMarker:
    def __init__(self, marked_sign):
        self.marked_sign = marked_sign

    def __getitem__(self, item):
        return self.marked_sign

    def is_marked(self, v):
        single_var = isinstance(v.val, py_defaults.Variable) and v.val.val is self
        return (v.val is self.marked_sign) or single_var


for_var_marker = ForVarMarker(Marker())


def for_stmt(target, iter_, body, orelse, modified_vars):

    target.val = for_var_marker

    with fresh_ast() as body_statements:
        body()

    for_vars = [
        Var(Ident(dmmy_rng, v.name))
        for v in modified_vars
        if for_var_marker.is_marked(v)
    ]

    for_vars = TupleLiteral(dmmy_rng, for_vars) if len(for_vars) > 1 else for_vars[0]

    node = For(dmmy_rng, [for_vars], [torch_expr(iter_)], body_statements)

    emit_node(node)


def if_stmt(cond, body, orelse, local_writes):
    c = cond()

    if isinstance(c, TorchExpr):
        with fresh_ast() as body_statements:
            body()
        with fresh_ast() as orelse_statements:
            orelse()

        emit_node(If(dmmy_rng, torch_expr(c), body_statements, orelse_statements))

    else:
        if c:
            body()
        else:
            orelse()


def and_(v1, v2):
    if isinstance(v1, bool) and isinstance(v2, bool):
        return v1 and v2

    if isinstance(v1, bool):
        if not v1:
            return False
        else:
            return v2

    if isinstance(v2, bool):
        if not v2:
            return False
        else:
            return v1

    return BinOp("and", v1, v2)


def return_(v):
    return flush_ast() + [Return(dmmy_rng, torch_expr(v))]


def wrap_ast(fn, statements):

    sourcelines, file_lineno = inspect.getsourcelines(fn)
    source = "".join(sourcelines)
    inspect.getsourcefile(fn)
    dedent_src = dedent(source)
    py_ast = ast.parse(dedent_src)

    param_list = torch.jit.frontend.build_param_list(ctx, py_ast.body[0].args, None)

    decl = Decl(dmmy_rng, param_list, None)
    definition = Def(Ident(dmmy_rng, fn.__code__.co_name), decl, statements)

    qual_name = fn.__module__ + "." + fn.__code__.co_name
    def_args = torch.jit.frontend.get_default_args(fn)

    closure_rcb = torch._jit_internal.createResolutionCallbackFromClosure(fn)
    stack_rcb = torch._jit_internal.createResolutionCallback(1)

    def _rcb(name):
        result = closure_rcb(name)
        if result:
            return result
        return stack_rcb(name)

    ret = torch._C._jit_script_compile(qual_name, definition, _rcb, def_args)

    return ret, definition


def specialize(fn, *args):
    converted_fn = conversion.convert(fn, tch_ir_, [variables, functions, control_flow])
    statements = converted_fn(*args)
    fn, ast = tch_ir_.wrap_ast(fn, statements)
    return fn, ast
