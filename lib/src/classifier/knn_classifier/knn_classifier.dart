import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/classifier/knn_classifier/_init_module.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory.dart';
import 'package:ml_algo/src/common/serializable/serializable.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/predictor/retrainable.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';

/// A class that performs classification basing on `k nearest neighbours` (KNN)
/// algorithm
///
/// K nearest neighbours algorithm is an algorithm that is targeted to search for
/// the most similar labelled observations (number of these observations is equal
/// to `k`) to the given unlabelled one.
///
/// It is possible to use majority class among the `k` found observations as a
/// prediction for the given unlabelled observation, but it may lead to the
/// imprecise result. Thus the weighted version of KNN algorithm is used in the
/// classifier. To get weight of a particular observation one may use a kernel
/// function.
abstract class KnnClassifier
    implements
        Assessable,
        Serializable,
        Retrainable,
        Classifier {
  /// Parameters:
  ///
  /// [trainData] Labelled observations. Must contain [targetName] column.
  ///
  /// [targetName] A string that serves as a name of the column containing
  /// outcomes.
  ///
  /// [k] a number of nearest neighbours to be found among [trainData]
  ///
  /// [kernel] a type of a kernel function that is used to predict an
  /// outcome for a new observation
  ///
  /// [distance] a distance type that is used to measure a distance between two
  /// observation vectors
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
        String classLabelPrefix = 'Class label',
        DType dtype = DType.float32,
      }
  ) => initKnnClassifierModule()
      .get<KnnClassifierFactory>()
      .create(
    trainData,
    targetName,
    k,
    kernel,
    distance,
    classLabelPrefix,
    dtype,
  );

  /// Restores previously fitted classifier instance from the given [json]
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
  /// final classifier = KnnClassifier(
  ///   samples,
  ///   targetName,
  ///   3,
  /// );
  ///
  /// final pathToFile = './classifier.json';
  ///
  /// await classifier.saveAsJson(pathToFile);
  ///
  /// final file = File(pathToFile);
  /// final json = await file.readAsString();
  /// final restoredClassifier = KnnClassifier.fromJson(json);
  ///
  /// // here you can use previously fitted restored classifier to make
  /// // some prediction, e.g. via `KnnClassifier.predict(...)`;
  /// ````
  factory KnnClassifier.fromJson(String json) =>
      initKnnClassifierModule()
          .get<KnnClassifierFactory>()
          .fromJson(json);

  /// A number of nearest neighbours
  ///
  /// The value is read-only, it's a hyperparameter of the model
  final int k;

  /// A kernel type
  ///
  /// The value is read-only, it's a hyperparameter of the model
  final KernelType kernelType;

  /// A distance type that is used to measure a distance between two
  /// observations
  ///
  /// The value is read-only, it's a hyperparameter of the model
  final Distance distanceType;
}
