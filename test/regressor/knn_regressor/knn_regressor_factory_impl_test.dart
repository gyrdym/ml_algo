import 'package:ml_algo/src/knn_solver/kernel_function/kernel_type.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory_impl.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('KnnRegressorFactoryImpl', () {
    final solverMock = KnnSolverMock();
    final kernelFnFactory = createKernelFactoryMock(($, [$$]) => null);
    final solverFnFactory = createKnnSolverFactoryMock(solverMock);

    final factory = KnnRegressorFactoryImpl(kernelFnFactory, solverFnFactory);

    final data = DataFrame.fromSeries(
        [
          Series('first' , <num>[1, 1, 1, 1]),
          Series('second', <num>[2, 2, 2, 2]),
          Series('third' , <num>[2, 2, 2, 2]),
          Series('fourth', <num>[4, 4, 4, 4]),
          Series('fifth' , <num>[50, 52, 53, 55]),
        ]
    );

    final targetName = 'fifth';

    test('should return a knn regressor', () {
      final regressor = factory.create(
        data,
        targetName,
        2,
        Kernel.epanechnikov,
        Distance.hamming,
        DType.float32,
      );

      verify(kernelFnFactory.createByType(Kernel.epanechnikov)).called(1);
      verify(solverFnFactory.create(
        argThat(equals([
          [1, 2, 2, 4],
          [1, 2, 2, 4],
          [1, 2, 2, 4],
          [1, 2, 2, 4],
        ])),
        argThat(equals([
          [50],
          [52],
          [53],
          [55],
        ])),
        2,
        Distance.hamming,
        true,
      )).called(1);

      expect(regressor, isA<KnnRegressorImpl>());
    });

    test('should throw an exception if target column does not exist in the '
        'train data', () {
      final actual = () => factory.create(
        data,
        'unknown_column',
        2,
        Kernel.uniform,
        Distance.hamming,
        DType.float32,
      );

      expect(actual, throwsException);
    });
  });
}
