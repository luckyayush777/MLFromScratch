I was reading an implementation of Adam and became curious, so I went backwards and decided to implement as many optimization procedures as possible on toy neural networks. This is my attempt to recreate the performance optimization measures that modern deep learning uses implicitly and exposes to us only through parameters.

We start with a hard-coded 2-layer perceptron and implement forward propagation and backpropagation. The implementation is kept loose enough to allow benchmarking, so I can observe what measures compilers take to make basic matrix multiplications fast.

These are not new models, and this project is not particularly useful beyond curiosity and my dislike of Python. Everything is written in minimal C++ (no heavy OOP, no shared pointers). The focus is on cache-friendly paradigms rather than abstracting everything into clean objects.

There is support for tensors (minimal) and image loading now. The point was to use the MNIST fashion dataset. Even with 2D tensors I should be able to do everything I want on this specific dataset.

Progress (brief):
- Forward/backward pass working for conv + MLP stack.
- Softmax cross-entropy verified with a perfect-prediction test.
- Basic conv/maxpool/flatten pipeline running end-to-end.

Build + data setup (Windows, CMake):
1) Download the MNIST files and place them in a top-level datasets folder:
	- train-images-idx3-ubyte
	- train-labels-idx1-ubyte
	- t10k-images-idx3-ubyte
	- t10k-labels-idx1-ubyte
2) Configure and build:
	- cmake -S . -B build
	- cmake --build build --config Debug
3) Run:
	- .\build\Debug\ml.exe

TODO:
- Dropout support
- Optimize the 7 inner loops
- Custom BLAS