import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory_impl.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_impl.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('KnnClassifierFactoryImpl', () {
    final solverMock = KnnSolverMock();
    final kernelMock = KernelMock();
    final kernelFnFactory = createKernelFactoryMock(kernelMock);
    final solverFnFactory = createKnnSolverFactoryMock(solverMock);

    final factory = KnnClassifierFactoryImpl(kernelFnFactory, solverFnFactory);

    final data = DataFrame.fromSeries(
      [
        Series('first' , <num>[1, 1, 1, 1]),
        Series('second', <num>[2, 2, 2, 2]),
        Series('third' , <num>[2, 2, 2, 2]),
        Series('fourth', <num>[4, 4, 4, 4]),
        Series('fifth' , <num>[1, 3, 2, 1], isDiscrete: true),
      ]
    );

    final targetName = 'fifth';

    test('should return a knn classifier', () {
      final classifier = factory.create(
        data,
        targetName,
        2,
        KernelType.uniform,
        Distance.hamming,
        DType.float32,
      );

      verify(kernelFnFactory.createByType(KernelType.uniform)).called(1);
      verify(solverFnFactory.create(
        argThat(equals([
          [1, 2, 2, 4],
          [1, 2, 2, 4],
          [1, 2, 2, 4],
          [1, 2, 2, 4],
        ])),
        argThat(equals([
          [1],
          [3],
          [2],
          [1],
        ])),
        2,
        Distance.hamming,
        true,
      )).called(1);

      expect(classifier, isA<KnnClassifierImpl>());
    });

    test('should extract class label list from target column even if the '
        'latter is not discrete', () {
      final data = DataFrame.fromSeries(
          [
            Series('first' , <num>[1, 1, 1, 1]),
            Series('target' , <num>[1, 3, 2, 1]),
          ]
      );

      final classifier = factory.create(
        data,
        'target',
        2,
        KernelType.uniform,
        Distance.hamming,
        DType.float32,
      );

      verify(kernelFnFactory.createByType(KernelType.uniform)).called(1);
      verify(solverFnFactory.create(
        argThat(equals([
          [1],
          [1],
          [1],
          [1],
        ])),
        argThat(equals([
          [1],
          [3],
          [2],
          [1],
        ])),
        2,
        Distance.hamming,
        true,
      )).called(1);

      expect(classifier, isA<KnnClassifierImpl>());
    });

    test('should throw an exception if target column does not exist in the '
        'train data', () {
      final actual = () => factory.create(
        data,
        'unknown_column',
        2,
        KernelType.uniform,
        Distance.hamming,
        DType.float32,
      );

      expect(actual, throwsException);
    });
  });
}
