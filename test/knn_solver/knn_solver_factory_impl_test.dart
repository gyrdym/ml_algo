import 'package:ml_algo/src/knn_solver/knn_solver_factory_impl.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_impl.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('KnnSolverFactoryImpl', () {
    test('should return a proper knn solver instance', () {
      final factory = const KnnSolverFactoryImpl();

      final features = Matrix.fromList([
        [1, 1, 1, 1]
      ]);
      final outcomes = Matrix.fromList([
        [1]
      ]);
      final k = 1;
      final distance = Distance.euclidean;
      final standardize = true;

      expect(
          factory.create(
            features,
            outcomes,
            k,
            distance,
            standardize,
          ),
          isA<KnnSolverImpl>());
    });
  });
}
