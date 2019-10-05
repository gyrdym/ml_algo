import 'package:ml_linalg/matrix.dart';

abstract class LinearOptimizer {
  /// Returns a coefficients of the equation of a hyperplane.
  ///
  /// Dimensions of the returning matrix:
  ///
  /// ````
  /// number_of_features x number_of_target_columns
  /// ````
  ///
  /// In other words, one column in the coefficients matrix describes its own
  /// dedicated target.
  ///
  /// Let's say, one has a dataset, consisting of features:
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
  /// coefficients will be like that:
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
  /// optimization iteration
  ///
  /// [isMinimizingObjective] should the solver find a maxima or minima
  Matrix findExtrema({
      Matrix initialCoefficients,
      bool isMinimizingObjective,
    });
}
