import torch
from torch._C._jit_tree_views import Apply, Attribute, Ident, Select, Var

from pyctr.examples.torchscript.dmmy import dmmy_rng
from pyctr.examples.torchscript.expression import TorchExpr, torch_expr
from pyctr.overloads import py_defaults, staging


def kwargs_to_attribute_list(kwargs):
    return [
        Attribute(Ident(dmmy_rng, name), torch_expr(kwargs[name])) for name in kwargs
    ]


def generate_fun_for(name):
    attrs = name.split(".")
    ident = Var(Ident(dmmy_rng, "torch"))
    for a in attrs:
        ident = Select(ident, Ident(dmmy_rng, a))

    def fn(args, kwargs):
        args = torch_expr(list(args))
        kwargs = kwargs_to_attribute_list(kwargs)
        return TorchExpr(Apply(ident, args, kwargs))

    return fn


def rgetattr(object, name):
    attrs = name.split(".")
    res = getattr(object, attrs[0])
    for a in attrs[1:]:
        res = getattr(object, a)
    return res


class TorchCallOverload:
    def __init__(self, names):
        self._registry = {}

        for name in names:
            fun = rgetattr(torch, name)
            self._registry[id(fun)] = generate_fun_for(name)


torch_call = TorchCallOverload(
    [
        "cat",
        "sigmoid",
        "transpose",
        "relu",
        "max",
        "tanh",
        "matmul",
        "mm",
        "add",
        "dropout",
    ]
)


builtin_call = staging.RewritingCallOverload(py_defaults.call)


def builtin_apply(name, args, kwargs):
    return TorchExpr(Apply(Var(Ident(dmmy_rng, name)), args, kwargs))


@builtin_call.replaces(range)
def range_(stop):
    return builtin_apply("range", [torch_expr(stop)], [])


@builtin_call.replaces(int)
def int_(e):
    return builtin_apply("int", [torch_expr(e)], [])


@builtin_call.replaces(len)
def len_(e):
    return builtin_apply("len", [torch_expr(e)], [])


def call(old_fun, args, kwargs):
    fun = torch_call._registry.get(id(old_fun), None)
    if fun:
        return fun(args, kwargs)
    return builtin_call(old_fun, args, kwargs)
