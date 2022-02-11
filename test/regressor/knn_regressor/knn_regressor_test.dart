import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/regressor/knn_regressor/_injector.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';
import '../../mocks.mocks.dart';

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
    final knnRegressor = MockKnnRegressor();
    final knnRegressorFactory = createKnnRegressorFactoryMock(knnRegressor);

    setUp(
      () => knnRegressorInjector
        ..clearAll()
        ..registerSingleton<KnnRegressorFactory>(() => knnRegressorFactory),
    );

    test('should call KnnRegressorFactory in order to create a regressor', () {
      final regressor = KnnRegressor(
        data,
        targetName,
        2,
        kernel: KernelType.epanechnikov,
        distance: Distance.cosine,
        dtype: DType.float64,
      );

      verify(knnRegressorFactory.create(data, targetName, 2,
              KernelType.epanechnikov, Distance.cosine, DType.float64))
          .called(1);

      expect(regressor, same(knnRegressor));
    });

    test('should persist hyperparameters', () {
      final k = 213;
      final kernel = KernelType.epanechnikov;
      final distance = Distance.cosine;

      when(knnRegressor.k).thenReturn(k);
      when(knnRegressor.kernelType).thenReturn(kernel);
      when(knnRegressor.distanceType).thenReturn(distance);

      final regressor = KnnRegressor(
        data,
        targetName,
        213,
        kernel: KernelType.epanechnikov,
        distance: Distance.cosine,
        dtype: DType.float64,
      );

      expect(regressor.k, k);
      expect(regressor.kernelType, kernel);
      expect(regressor.distanceType, distance);
    });
  });
}
