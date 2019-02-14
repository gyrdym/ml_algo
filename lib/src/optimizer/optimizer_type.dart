/// A type of optimizer. Optimizer - is an algorithm that finds the best coefficients (or weights) for the passed
/// features (points)
/// [OptimizerType.gradientDescent] Gradient descent optimizer. On each iteration calculates a gradient vector of a cost
/// function to find the further direction towards the optimal point (maximum or minimum) along the function
/// [OptimizerType.coordinateDescent] Coordinate descent optimizer. Excludes a feature on each iteration and evaluates
/// a value of a cost function without this excluded feature. In other words, goes through iterations coordinate by
/// coordinate (term `feature` in this case is the same as `coordinate`). Allows to apply L1 regularization, since
/// there is no need in finding gradient vector
enum OptimizerType {
  gradientDescent,
  coordinateDescent,
}
