import 'package:injector/injector.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_type.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('KnnRegressor', () {
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

    final knnRegressor = KnnRegressorMock();
    final knnRegressorFactory = createKnnRegressorFactoryMock(knnRegressor);

    setUp(() => injector = Injector()
      ..registerSingleton<KnnRegressorFactory>((_) => knnRegressorFactory),
    );

    tearDown(() => injector = null);

    test('should call KnnRegressorFactory in order to create a regressor', () {
      final regressor = KnnRegressor(
        data,
        targetName,
        2,
        kernel: KernelType.epanechnikov,
        distance: Distance.cosine,
        dtype: DType.float64,
      );

      verify(knnRegressorFactory.create(
          data,
          targetName,
          2,
          KernelType.epanechnikov,
          Distance.cosine,
          DType.float64
      )).called(1);

      expect(regressor, same(knnRegressor));
    });
  });
}
