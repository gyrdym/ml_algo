/// A types of cost functions, which are used by different linear optimizers.
enum CostFunctionType {
  /// A logarithmic form of likelihood function.
  logLikelihood,

  /// A squared difference between actual and predicted values.
  squared,
}
