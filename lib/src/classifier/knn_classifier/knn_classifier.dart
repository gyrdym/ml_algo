import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/classifier/knn_classifier/_helpers/create_knn_classifier.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';

/// A class that performs classification basing on `k nearest neighbours` (KNN)
/// algorithm
///
/// K nearest neighbours algorithm is an algorithm that is targeted to search
/// most similar labelled observations (number of these observations equals `k`)
/// for the given unlabelled one.
///
/// It is possible to use majority class among the k found observations as a
/// prediction for the given unlabelled observation, but it may lead to the
/// imprecise result. Thus the weighted version of KNN algorithm is used in the
/// classifier. To get weight of a particular observation one may use a kernel
/// function.
abstract class KnnClassifier implements Assessable, Classifier {
  /// Parameters:
  ///
  /// [trainData] Labelled observations, among which will be searched [k]
  /// nearest to the given unlabelled observations neighbours. Must contain
  /// [targetName] column.
  ///
  /// [targetName] A string, that serves as a name of the column, that contains
  /// labels (or outcomes).
  ///
  /// [k] a number of nearest neighbours to be found among [trainData]
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
  factory KnnClassifier(
      DataFrame trainData,
      String targetName,
      int k,
      {
        KernelType kernel = KernelType.gaussian,
        Distance distance = Distance.euclidean,
        DType dtype = DType.float32,
      }
  ) => createKnnClassifier(trainData, targetName, k, kernel, distance, dtype);
}
