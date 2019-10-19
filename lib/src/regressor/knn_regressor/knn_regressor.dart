import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_impl.dart';
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
abstract class KnnRegressor implements Assessable, Predictor {
  /// Parameters:
  ///
  /// [fittingData] Labelled observations, among which will be searched [k]
  /// nearest neighbours for unlabelled observations. Must contain [targetName]
  /// column.
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
      int k,
      {
        Kernel kernel = Kernel.gaussian,
        Distance distance = Distance.euclidean,
        DType dtype = DType.float32,
      }) {
    final splits = featuresTargetSplit(fittingData,
      targetNames: [targetName],
    ).toList();

    return KnnRegressorImpl(
      splits[0].toMatrix(),
      splits[1].toMatrix(),
      targetName,
      k: k,
      kernel: kernel,
      distance: distance,
      dtype: dtype,
    );
  }
}
