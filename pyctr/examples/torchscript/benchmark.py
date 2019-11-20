from timeit import timeit

import torch
from torch.jit.frontend import get_jit_def

from pyctr.examples.torchscript import torchscript


def trace_rnn_w_dropout(w, seq, training):
    out = seq[0]  # initial state is first step
    bool_train = bool((training == 1).any())
    for i in range(len(seq) - 1):
        h = torch.cat((seq[i], out), dim=1)
        out = torch.tanh(h @ w)
        if bool_train:
            out = torch.dropout(out, 0.2, True)

    out = torch.tanh(torch.cat((seq[-1], out), dim=1) @ w)
    return out


def rnn_w_dropout(w, seq, train: bool):
    out = seq[0]
    for i in range(len(seq) - 1):
        h = torch.cat((seq[i], out), 1)
        out = torch.tanh(h @ w)
        if train:
            out = torch.dropout(out, 0.2, True)
    out = torch.tanh(torch.cat((seq[-1], out), 1) @ w)
    return out


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
    trace_training = torch.tensor([1])
    dev = "cpu"

    seq = torch.randn(seq_length, batch_size, input_dim).to(dev)
    w = torch.randn([state_cat_input_dim, state_dim], requires_grad=True).to(dev)
    expected = torch.ones(state_dim).to(dev)

    specialized_rnn, _ = torchscript.specialize(rnn_w_dropout, w, seq, training)
    scripted_rnn = torch.jit.script(rnn_w_dropout)
    traced_rnn = torch.jit.trace(trace_rnn_w_dropout, (w, seq, trace_training))

    vanilla_time = timeit(
        lambda: full_bench(rnn_w_dropout, (w, seq, training), expected), number=n_times
    )
    specialize_time = timeit(
        lambda: full_bench(specialized_rnn, (w, seq, training), expected),
        number=n_times,
    )
    script_time = timeit(
        lambda: full_bench(scripted_rnn, (w, seq, training), expected), number=n_times
    )

    trace_time = timeit(
        lambda: full_bench(traced_rnn, (w, seq, trace_training), expected),
        number=n_times,
    )

    print(f"Running forward + backward benchmark (training={training})")
    print(f"Time vanilla {vanilla_time}")
    print(f"Time `pyctr.specialize` {specialize_time}")
    print(f"Time `torch.jit.script` {script_time}")
    print(f"Time `torch.jit.trace` {trace_time}")

    print(
        f"Time wrt script: vanilla {script_time / vanilla_time}, specialize {script_time / specialize_time}, trace {script_time / trace_time}"
    )
