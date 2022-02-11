import 'package:ml_linalg/matrix.dart';

abstract class LinearOptimizer {
  /// Returns coefficients of the equation of a hyperplane.
  ///
  /// Dimensions of the returning matrix:
  ///
  /// ````
  /// number_of_features x number_of_target_columns
  /// ````
  ///
  /// In other words, one column in the coefficient matrix describes its own
  /// dedicated target.
  ///
  /// Let's say, one has a dataset consisting of the features:
  ///
  /// ````dart
  /// final x = [
  ///   [10, 20, 30],
  ///   [11, 22, 33],
  ///   [22, 33, 43],
  ///   [89, 76, 32],
  /// ];
  /// ````
  ///
  /// And outcomes:
  ///
  /// ````dart
  /// final y = [
  ///   [100],
  ///   [200],
  ///   [300],
  ///   [400],
  /// ];
  /// ````
  ///
  /// After solving the equation of a hyperplane via the [LinearOptimizer], the
  /// coefficients will be like that (values using below are random, pay
  /// attention just to the shape of the matrix):
  ///
  /// ````dart
  /// [
  ///   [1.5],
  ///   [0.3],
  ///   [2.4],
  /// ]
  /// ````
  ///
  /// Parameters:
  ///
  /// [initialCoefficients] initial coefficients that will be used in the first
  /// optimization iteration. Meaningless in case of closed-form solution.
  ///
  /// [isMinimizingObjective] should the solver find a maxima or minima.
  /// Meaningless in case of closed-form solution.
  ///
  /// [collectLearningData] whether or not to collect learning-related data,
  /// such as errors from cost function, after every iteration. May affect
  /// performance. Meaningless in case of closed-form solution.
  Matrix findExtrema({
    Matrix? initialCoefficients,
    bool isMinimizingObjective = true,
    bool collectLearningData = false,
  });

  /// Returns a list of errors from a cost function after every learning
  /// iteration
  List<num> get costPerIteration;
}
