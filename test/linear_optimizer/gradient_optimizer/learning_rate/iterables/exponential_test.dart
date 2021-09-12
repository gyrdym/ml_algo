import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/iterables/exponential.dart';
import 'package:test/test.dart';

import '../../../../helpers.dart';

void main() {
  group('ExponentialLearningRateIterable', () {
    test('should return a correct sequence, limit=4', () {
      final elements = ExponentialLearningRateIterable(
          initialValue: 1.3, decay: 2, limit: 4);

      expect(
          elements, iterableAlmostEqualTo([0.175, 0.023, 0.003, 0.0004], 1e-3));
    });

    test('should return a correct sequence, limit=2', () {
      final elements = ExponentialLearningRateIterable(
          initialValue: 1.3, decay: 2, limit: 2);

      expect(elements, iterableAlmostEqualTo([0.175, 0.023], 1e-3));
    });

    test('should return a correct sequence, limit=0', () {
      final elements = ExponentialLearningRateIterable(
          initialValue: 1.3, decay: 2, limit: 0);

      expect(elements, iterableAlmostEqualTo([], 1e-3));
    });
  });
}
