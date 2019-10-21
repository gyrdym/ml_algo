import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';

abstract class KnnClassifier implements Classifier {
  factory KnnClassifier(
      DataFrame fittingData,
      String targetName,
      int k,
      {
        KernelType kernel = KernelType.gaussian,
        Distance distance = Distance.euclidean,
        DType dtype = DType.float32,
      }
  ) => dependencies
      .getDependency<KnnClassifierFactory>()
      .create(fittingData, targetName, k, kernel, distance, dtype);
}
