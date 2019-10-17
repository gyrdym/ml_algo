import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory.dart';

class KnnSolverFactoryImpl implements KnnSolverFactory {
  const KnnSolverFactoryImpl();

  @override
  FindKnnFn create() => findKNeighbours;
}
