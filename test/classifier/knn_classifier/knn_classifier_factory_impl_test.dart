import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory_impl.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_impl.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('KnnClassifierFactoryImpl', () {
    final kernelFnFactory = createKernelFactoryMock(($, [$$]) => null);
    final solverFnFactory = createKnnSolverFactoryMock(
        ($, $$, $$$, $$$$, {distance, standardize}) => null);

    final factory = KnnClassifierFactoryImpl(kernelFnFactory, solverFnFactory);

    final data = DataFrame(
      <Iterable<num>>[
        [1, 2, 2, 4, 5],
        [1, 2, 2, 4, 5],
        [1, 2, 2, 4, 5],
        [1, 2, 2, 4, 5],
      ],
      headerExists: false,
      header: ['first', 'second', 'third', 'fourth', 'fifth'],
    );

    final targetName = 'fifth';

    test('should return a knn classifier', () {
      final classifier = factory.create(
        data,
        targetName,
        2,
        Kernel.uniform,
        Distance.euclidean,
        DType.float32,
      );

      verify(kernelFnFactory.createByType(Kernel.uniform)).called(1);
      verify(solverFnFactory.create()).called(1);

      expect(classifier, isA<KnnClassifierImpl>());
    });
  });
}
