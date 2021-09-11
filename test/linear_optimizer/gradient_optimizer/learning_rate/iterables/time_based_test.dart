import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/iterables/time_based.dart';
import 'package:test/test.dart';

import '../../../../helpers.dart';

void main() {
  group('TimeBasedLearningRateIterable', () {
    test('should return a correct sequence, limit=4', () {
      final elements =
          TimeBasedLearningRateIterable(initialValue: 1.3, decay: 2, limit: 4);

      expect(
          elements, iterableAlmostEqualTo([0.433, 0.086, 0.012, 0.001], 1e-3));
    });

    test('should return a correct sequence, limit=2', () {
      final elements =
          TimeBasedLearningRateIterable(initialValue: 1.3, decay: 2, limit: 2);

      expect(elements, iterableAlmostEqualTo([0.433, 0.086], 1e-3));
    });

    test('should return a correct sequence, limit=0', () {
      final elements =
          TimeBasedLearningRateIterable(initialValue: 1.3, decay: 2, limit: 0);

      expect(elements, iterableAlmostEqualTo([], 1e-3));
    });
  });
}
