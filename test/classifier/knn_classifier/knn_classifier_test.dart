import 'package:injector/injector.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('KnnClassifier', () {
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

    final knnClassifier = KnnClassifierMock();
    final knnClassifierFactory = createKnnClassifierFactoryMock(knnClassifier);

    setUp(() => injector = Injector()
      ..registerSingleton<KnnClassifierFactory>((_) => knnClassifierFactory),
    );

    tearDown(() => injector = null);

    test('should call KnnClassifierFactory in order to create a classifier', () {
      final classifier = KnnClassifier(
        data,
        targetName,
        2,
        kernel: KernelType.uniform,
        distance: Distance.cosine,
        dtype: DType.float64,
      );

      verify(knnClassifierFactory.create(
          data,
          targetName,
          2,
          KernelType.uniform,
          Distance.cosine,
          DType.float64
      )).called(1);

      expect(classifier, same(knnClassifier));
    });
  });
}
