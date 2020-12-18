import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory_impl.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_impl.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('KnnClassifierFactoryImpl', () {
    test('should create a KnnClassifierImpl instance', () {
      final factory = const KnnClassifierFactoryImpl();
      final kernelMock = KernelMock();
      final solverMock = KnnSolverMock();
      final dtype = DType.float32;

      final actual = factory.create(
        'target',
        [1, 2, 3],
        kernelMock,
        solverMock,
        'label',
        dtype,
      );

      expect(actual, isA<KnnClassifierImpl>());
    });
  });
}
