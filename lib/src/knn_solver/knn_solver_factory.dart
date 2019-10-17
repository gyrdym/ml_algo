import 'package:ml_algo/src/knn_solver/knn_solver.dart';

abstract class KnnSolverFactory {
  FindKnnFn create();
}
