enum LinearOptimizerType {
  /// Original gradient descent optimization, only L2 regularization is
  /// applicable while optimizing a function via this method
  vanillaGD,

  /// Original coordinate descent optimization, only L1 regularization is
  /// applicable while optimizing a function via this method
  vanillaCD,
}
