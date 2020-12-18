import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier.dart';
import 'package:ml_algo/src/knn_kernel/kernel.dart';
import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_linalg/dtype.dart';

abstract class KnnClassifierFactory {
  KnnClassifier create(
      String targetName,
      List<num> classLabels,
      Kernel kernel,
      KnnSolver solver,
      String columnPrefix,
      DType dtype,
  );
}
