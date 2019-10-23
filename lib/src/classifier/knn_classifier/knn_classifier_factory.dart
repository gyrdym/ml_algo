import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';

abstract class KnnClassifierFactory {
  KnnClassifier create(
      DataFrame fittingData,
      String targetName,
      int k,
      KernelType kernelType,
      Distance distance,
      DType dtype,
  );
}
