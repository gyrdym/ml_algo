import 'package:ml_algo/src/common/serializable/serializable.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_algo/src/predictor/retrainable.dart';
import 'package:ml_algo/src/regressor/knn_regressor/_init_module.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';

/// A class that performs regression basing on `k nearest neighbours` algorithm
///
/// K nearest neighbours algorithm is an algorithm that is targeted to search
/// most similar labelled observations (number of these observations equals `k`)
/// for the given unlabelled one.
///
/// In order to make a prediction, or rather to set a label for a given new
/// observation, labels of found `k` observations are being summed up and
/// divided by `k`.
///
/// To get a more precise result, one may use weighted average of found labels -
/// the farther a found observation from the target one, the lower the weight of
/// the observation is. To obtain these weights one may use a kernel function.
abstract class KnnRegressor
    implements Assessable, Serializable, Retrainable<KnnRegressor>, Predictor {
  /// Parameters:
  ///
  /// [fittingData] Labelled observations, among which will be searched [k]
  /// nearest to the given unlabelled observations neighbours. Must contain
  /// [targetName] column.
  ///
  /// [targetName] A string, that serves as a name of the column, that contains
  /// labels (or outcomes).
  ///
  /// [k] a number of nearest neighbours to be found among [fittingData]
  ///
  /// [kernel] a type of a kernel function, that will be used to predict an
  /// outcome for a new observation
  ///
  /// [distance] a distance type, that will be used to measure a distance
  /// between two observation vectors
  ///
  /// [dtype] A data type for all the numeric values, used by the algorithm. Can
  /// affect performance or accuracy of the computations. Default value is
  /// [DType.float32]
  factory KnnRegressor(
    DataFrame fittingData,
    String targetName,
    int k, {
    KernelType kernel = KernelType.gaussian,
    Distance distance = Distance.euclidean,
    DType dtype = DType.float32,
  }) =>
      initKnnRegressorModule().get<KnnRegressorFactory>().create(
            fittingData,
            targetName,
            k,
            kernel,
            distance,
            dtype,
          );

  /// Restores previously fitted regressor instance from the given [json]
  ///
  /// ````dart
  /// import 'dart:io';
  /// import 'package:ml_dataframe/ml_dataframe.dart';
  ///
  /// final data = <Iterable>[
  ///   ['feature 1', 'feature 2', 'feature 3', 'outcome']
  ///   [        5.0,         7.0,         6.0,       1.0],
  ///   [        1.0,         2.0,         3.0,       0.0],
  ///   [       10.0,        12.0,        31.0,       0.0],
  ///   [        9.0,         8.0,         5.0,       0.0],
  ///   [        4.0,         0.0,         1.0,       1.0],
  /// ];
  /// final targetName = 'outcome';
  /// final samples = DataFrame(data, headerExists: true);
  /// final regressor = KnnRegressor(
  ///   samples,
  ///   targetName,
  ///   3,
  /// );
  ///
  /// final pathToFile = './regressor.json';
  ///
  /// await regressor.saveAsJson(pathToFile);
  ///
  /// final file = File(pathToFile);
  /// final json = await file.readAsString();
  /// final restoredRegressor = KnnRegressor.fromJson(json);
  ///
  /// // here you can use previously fitted restored regressor to make
  /// // some prediction, e.g. via `restoredRegressor.predict(...)`;
  /// ````
  factory KnnRegressor.fromJson(String json) =>
      initKnnRegressorModule().get<KnnRegressorFactory>().fromJson(json);

  /// A number of nearest neighbours
  ///
  /// The value is read-only, it's a hyperparameter of the model
  int get k;

  /// A kernel type
  ///
  /// The value is read-only, it's a hyperparameter of the model
  KernelType get kernelType;

  /// A distance type that is used to measure a distance between two
  /// observations
  ///
  /// The value is read-only, it's a hyperparameter of the model
  Distance get distanceType;
}
