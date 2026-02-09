struct Conv2d {
  int inChannels;
  int outChannels;
  int kernel;
  int stride;
  int padding;

  // W : [outChannels, inChannels, kernel, kernel]
  Tensor W;
  Tensor b;

  Tensor X_cache; // [N, C, H, W]

  Conv2d(size_t outCh, size_t k, size_t stride, size_t padding)
      : inChannels(1), outChannels(outCh), kernel(k), stride(stride),
        padding(padding), W({outCh, k, k}), b({outCh}) {
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 0.01);
    for (size_t i = 0; i < W.noOfElements(); ++i)
      W.flat(i) = dist(rng);
    for (size_t i = 0; i < b.noOfElements(); ++i)
      b.flat(i) = 0.0;
  }

  double getPaddedInput(const Tensor &X, size_t batch, size_t channel, int h,
                        int w);
  Tensor conv2dForward(const Conv2d &conv, const Tensor &input);
  Tensor flattenForward(const Tensor &input);
  Tensor maxPool2dForward(const Tensor &input, size_t poolSize, size_t stride);
};