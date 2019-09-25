import 'package:ml_algo/src/algorithms/knn/kernel_type.dart';
import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/regressor/knn_regressor.dart';
import 'package:ml_algo/src/regressor/regressor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';

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
  factory ParameterlessRegressor.knn(DataFrame fittingData, {
    int targetIndex,
    String targetName,
    int k,
    Kernel kernel = Kernel.uniform,
    Distance distance = Distance.euclidean,
    DType dtype = DType.float32,
  }) {
    final featuresTargetSplits = featuresTargetSplit(fittingData,
      targetIndices: [targetIndex],
      targetNames: [targetName],
    ).toList();

    return KNNRegressor(
      featuresTargetSplits[0].toMatrix(),
      featuresTargetSplits[1].toMatrix(),
      k: k,
      kernel: kernel,
      distance: distance,
      dtype: dtype,
    );
  }
}
