import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_factory_impl.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_impl.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

import '../../../mocks.dart';

void main() {
  group('LogisticRegressorFactoryImpl', () {
    test('should create a LogisticRegressorImpl instance', () {
      final factory = const LogisticRegressorFactoryImpl();

      final targetName = 'target';
      final linkFunction = LinkFunctionMock();
      final probabilityThreshold = 0.7;
      final fitIntercept = false;
      final interceptScale = 10;
      final coefficients = Matrix.fromList([
        [1],
        [2],
        [3],
      ]);
      final negativeLabel = 0;
      final positiveLabel = 1;
      final dtype = DType.float32;

      final actual = factory.create(targetName, linkFunction,
          probabilityThreshold, fitIntercept, interceptScale,
          coefficients, negativeLabel, positiveLabel, dtype);

      expect(actual, isA<LogisticRegressorImpl>());
    });
  });
}
