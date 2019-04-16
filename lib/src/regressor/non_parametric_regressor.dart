import 'package:ml_algo/src/regressor/knn_regressor.dart';
import 'package:ml_algo/src/regressor/regressor.dart';
import 'package:ml_linalg/distance.dart';

/// A factory for all the non parametric family of Machine Learning algorithms
abstract class NoNParametricRegressor implements Regressor {
  /// Creates an instance of KNN regressor
  /// KNN here means "K nearest neighbor"
  /// [k] a number of neighbors
  factory NoNParametricRegressor.nearestNeighbor({
    int k,
    Distance distanceType,
  }) = KNNRegressor;
}
