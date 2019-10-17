import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_factory_impl.dart';
import 'package:test/test.dart';

void main() {
  group('KnnSolverFactoryImpl', () {
    test('should return appropriate solver function', () {
      final factory = const KnnSolverFactoryImpl();
      expect(factory.create(), same(findKNeighbours));
    });
  });
}
