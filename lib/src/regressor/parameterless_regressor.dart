import 'package:ml_algo/src/algorithms/knn/kernel_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/regressor/knn_regressor.dart';
import 'package:ml_algo/src/regressor/regressor.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/matrix.dart';

/// A factory for all the non parametric family of Machine Learning algorithms
abstract class ParameterlessRegressor implements Regressor, Assessable {
  /// Creates an instance of KNN regressor
  ///
  /// KNN here means "K nearest neighbor"
  ///
  /// [k] a number of nearest neighbours
  ///
  /// [kernel] a type of kernel function, that will be used to find an outcome
  /// for a new observation
  ///
  /// [distance] a distance type, that will be used to measure a distance
  /// between two observation vectors
  factory ParameterlessRegressor.knn(Matrix trainingFeatures,
      Matrix trainingOutcomes, {
    int k,
    Kernel kernel,
    Distance distance,
  }) = KNNRegressor;
}
