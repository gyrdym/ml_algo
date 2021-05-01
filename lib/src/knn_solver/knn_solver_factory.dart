import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/matrix.dart';

abstract class KnnSolverFactory {
  KnnSolver create(
    Matrix trainFeatures,
    Matrix trainLabels,
    int k,
    Distance distanceType,
    bool standardize,
  );
}
