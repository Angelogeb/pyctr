from timeit import timeit

import torch
from torch.jit.frontend import get_jit_def

from pyctr.examples.torchscript import torchscript


def foo(w, seq, training: bool):
    ret = seq[0]  # initial state is first step
    for i in range(len(seq) - 1):
        h = torch.cat((seq[i], ret), dim=1)
        ret = torch.tanh(h @ w)
        if training:
            ret = torch.dropout(ret, 0.5, training)

    ret = torch.tanh(torch.cat((seq[-1], ret), dim=1) @ w)
    return ret


def forward_bench(fun, args, expected):
    fun(*args)


def full_bench(fun, args, expected):
    predicted = fun(*args)
    loss = ((predicted - expected) ** 2).sum()
    loss.backward()


if __name__ == "__main__":
    batch_size = 512
    seq_length = 1000
    state_dim = 64
    input_dim = 64
    n_times = 10
    state_cat_input_dim = state_dim + input_dim
    training = True
    dev = "cpu"

    seq = torch.randn(seq_length, batch_size, input_dim).to(dev)
    w = torch.randn([state_cat_input_dim, state_dim], requires_grad=True).to(dev)

    specialized_foo, _ = torchscript.specialize(foo, w, seq, training)
    jitted_foo = torch.jit.script(foo)

    expected = torch.ones(state_dim).to(dev)

    print(f"Running forward + backward benchmark (training={training})")
    print(
        f"Time vanilla {timeit(lambda: full_bench(foo, (w, seq, training), expected), number=n_times)}"
    )
    print(
        f"Time `torch.jit.script` {timeit(lambda: full_bench(jitted_foo, (w, seq, training), expected), number=n_times)}"
    )
    print(
        f"Time `pyctr.specialize` {timeit(lambda: full_bench(specialized_foo, (w, seq, training), expected), number=n_times)}"
    )


    print(
        f"Time vanilla {timeit(lambda: full_bench(foo, (w, seq, training), expected), number=n_times)}"
    )
    print(
        f"Time `torch.jit.script` {timeit(lambda: full_bench(jitted_foo, (w, seq, training), expected), number=n_times)}"
    )
    print(
        f"Time `pyctr.specialize` {timeit(lambda: full_bench(specialized_foo, (w, seq, training), expected), number=n_times)}"
    )

