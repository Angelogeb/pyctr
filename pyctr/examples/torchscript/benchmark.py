from timeit import timeit

import torch
from torch.jit.frontend import get_jit_def

from pyctr.examples.torchscript import torchscript


def foo(w, seq, training: bool):
    ret = seq[0]  # initial state is first step
    for i in range(len(seq) - 1):
        ret = torch.tanh(w @ torch.cat((seq[i], ret), dim=0))
        if training:
            ret = torch.dropout(ret, 0.5, training)

    ret = torch.tanh(w @ torch.cat((seq[-1], ret), dim=0))
    return ret


def full_bench(fun, args, expected):
    predicted = fun(*args)
    loss = ((predicted - expected) ** 2).sum()
    loss.backward()


if __name__ == "__main__":
    batch_size = 30
    seq_length = 1000
    state_dim = 20
    input_dim = 20
    state_cat_input_dim = state_dim + input_dim
    training = True

    seq = torch.randn(seq_length, input_dim)
    w = torch.randn([state_dim, state_cat_input_dim], requires_grad=True)

    foo(w, seq, training)

    specialized_foo, _ = torchscript.specialize(foo, w, seq, training)
    jitted_foo = torch.jit.script(foo)

    expected = torch.ones(state_dim)

    print(
        f"Time jitted {timeit(lambda: full_bench(jitted_foo, (w, seq, training), expected), number=100)}"
    )
    print(
        f"Time specialized {timeit(lambda: full_bench(specialized_foo, (w, seq, training), expected), number=100)}"
    )

    print(
        f"Time jitted {timeit(lambda: full_bench(jitted_foo, (w, seq, training), expected), number=100)}"
    )
    print(
        f"Time specialized {timeit(lambda: full_bench(specialized_foo, (w, seq, training), expected), number=100)}"
    )

    print(
        f"Time jitted {timeit(lambda: full_bench(jitted_foo, (w, seq, training), expected), number=100)}"
    )
    print(
        f"Time specialized {timeit(lambda: full_bench(specialized_foo, (w, seq, training), expected), number=100)}"
    )
