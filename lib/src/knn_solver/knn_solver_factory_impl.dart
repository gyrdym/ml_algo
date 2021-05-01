import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_impl.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/matrix.dart';

class KnnSolverFactoryImpl implements KnnSolverFactory {
  const KnnSolverFactoryImpl();

  @override
  KnnSolver create(
    Matrix trainFeatures,
    Matrix trainLabels,
    int k,
    Distance distanceType,
    bool standardize,
  ) =>
      KnnSolverImpl(
        trainFeatures,
        trainLabels,
        k,
        distanceType,
        standardize,
      );
}
