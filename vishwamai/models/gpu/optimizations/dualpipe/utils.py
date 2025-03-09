"""
MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import queue
from typing import List, Callable

import torch
from torch.autograd import Variable


class WeightGradStore:

    enabled: bool = False
    cache: List[Callable] = []
    funcs_queue = queue.Queue()

    @classmethod
    def put(cls, func: Callable) -> None:
        cls.cache.append(func)

    @classmethod
    def flush(cls) -> None:
        cls.funcs_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls) -> None:
        assert not cls.funcs_queue.empty(), "Pop empty queue."
        funcs = cls.funcs_queue.get()
        for func in funcs:
            func()

    @classmethod
    def clear(cls) -> None:
        cls.cache = []
        cls.funcs_queue = queue.Queue()


def run_backward(tensors: List[torch.Tensor], grad_tensors: List[torch.Tensor]) -> None:
    kwargs = dict(
        keep_graph=False,
        create_graph=False,
        allow_unreachable=True,
        accumulate_grad=True,
    )
    Variable._execution_engine.run_backward(tensors, grad_tensors, **kwargs)


def chunk_tensor(x, chunks, dim):
    if x is None:
        return [None for _ in range(chunks)]
    return x.tensor_split(chunks, dim=dim)


def cat_tensor(x, dim):
    if (isinstance(x, tuple) or isinstance(x, list)):
        if len(x) == 1:
            return x[0]
        elif x[0] is None:
            assert all(y is None for y in x)
            return None
    return torch.cat(x, dim=dim)


def scatter(inputs, chunks, dim):
    assert isinstance(inputs, (torch.Tensor, tuple, list))
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    assert all(x is None or isinstance(x, torch.Tensor) for x in inputs)
    inputs = [chunk_tensor(x, chunks, dim) for x in inputs]
    microbatches = [microbatch for microbatch in zip(*inputs)]
    if len(microbatches) == 0:
        microbatches = [() for _ in range(chunks)]
    return microbatches


def gather(micro_outputs, dim):
    assert isinstance(micro_outputs[0], (torch.Tensor, tuple, list))
    if isinstance(micro_outputs[0], torch.Tensor):
        micro_outputs = [(x,) for x in micro_outputs]
    outputs = [x for x in zip(*micro_outputs)]
    outputs = tuple(cat_tensor(x, dim=dim) for x in outputs)
    return outputs
