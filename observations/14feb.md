## Built a CNN from scratch

Built a convolutional neural network from scratch.

A convolution is such a ridiculously beautiful mathematical operation. The first time I really understood it, it genuinely blew my mind. You’re not changing the core nature of a function — you’re changing how it behaves locally. You can smooth it, sharpen it, bias it toward certain structures. Same underlying function, different “texture”. That’s just cool.

It also shows up everywhere. Signal processing, probability, PDEs, vision. I was reading about its use with FFTs and how convolution becomes multiplication in frequency space. That part is still a bit beyond me, but it’s fascinating.

---

## Model

Very simple architecture:

- Input  
- Convolution  
- Max pooling  
- ReLU  
- Softmax  
- One MLP layer (basically just a fully connected layer)

I also implemented basic velocity (momentum-style updates). It actually gives non-insignificant gains during training, which was nice to see.

---

## The scary part

The convolution implementation has seven nested loops.

Seven.

At some point you just accept your fate and start indexing carefully.

It’s not intellectually hard. It’s just rigorous in an annoying way:
- tracking shapes
- channels
- strides
- padding
- output dimensions
- not messing up offsets

Most books absolutely skip the painful parts.

I keep reminding myself: even fancy libraries ultimately do this somewhere. They just:
- tile it
- parallelize it
- use SIMD
- transform to GEMM
- or use FFT tricks

But under the hood, someone is still multiplying and accumulating numbers in loops.

---

## Performance notes

Switched to raw C-style pointers and got ~20% speedup.

Turns out:
- the CPU does not like exceptions in hot paths
- it does not like bounds checking inside seven inner loops
- tiny overhead × millions of iterations = real slowdown

So yeah. Safety vs performance is very real when you’re inside numerical kernels.

Maybe exceptions aren’t ideal here. Future me problem.

---

## Weird issue

Backward propagation time increases per epoch (in debug config).

Forward time stays stable.
Accuracy is fine.

That feels like:
- a memory leak
- or some repeated allocation in backward
- or debug instrumentation compounding

It’s weird. Needs digging.

---

## Still slow

Obviously still slow.

At some point I want to try writing a tiny BLAS just to understand what that world feels like.

The math isn’t hard.
The implementation is annoying.
The performance tuning is humbling.

But when it works, it feels ridiculously satisfying.
