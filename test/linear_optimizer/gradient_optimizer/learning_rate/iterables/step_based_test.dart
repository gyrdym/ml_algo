import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/iterables/step_based.dart';
import 'package:test/test.dart';

import '../../../../helpers.dart';

void main() {
  group('StepBasedLearningRateIterable', () {
    test('should return a correct sequence, limit=4, dropRate=2', () {
      final elements =
      StepBasedLearningRateIterable(initialValue: 1.3, decay: 2, dropRate: 2, limit: 4);

      expect(
          elements, iterableAlmostEqualTo([2.6, 2.6, 5.2, 5.2], 1e-2));
    });

    test('should return a correct sequence, limit=4, dropRate=1', () {
      final elements =
      StepBasedLearningRateIterable(initialValue: 1.3, decay: 2, dropRate: 1, limit: 4);

      expect(
          elements, iterableAlmostEqualTo([5.2, 10.4, 20.8, 41.6], 1e-2));
    });

    test('should return a correct sequence, limit=4, decay=-2, dropRate=1', () {
      final elements =
      StepBasedLearningRateIterable(initialValue: 1.3, decay: -2, dropRate: 1, limit: 4);

      expect(
          elements, iterableAlmostEqualTo([5.2, -10.4, 20.8, -41.6], 1e-2));
    });

    test('should return a correct sequence, limit=3, dropRate=2', () {
      final elements =
      StepBasedLearningRateIterable(initialValue: 1.3, decay: 2, dropRate: 2, limit: 3);

      expect(
          elements, iterableAlmostEqualTo([2.6, 2.6, 5.2], 1e-2));
    });

    test('should return a correct sequence, limit=2, dropRate=2', () {
      final elements =
      StepBasedLearningRateIterable(initialValue: 1.3, decay: 2, dropRate: 2, limit: 2);

      expect(
          elements, iterableAlmostEqualTo([2.6, 2.6], 1e-2));
    });

    test('should return a correct sequence, limit=1, dropRate=2', () {
      final elements =
      StepBasedLearningRateIterable(initialValue: 1.3, decay: 2, dropRate: 2, limit: 1);

      expect(
          elements, iterableAlmostEqualTo([2.6], 1e-2));
    });

    test('should return a correct sequence, limit=0, dropRate=2', () {
      final elements =
      StepBasedLearningRateIterable(initialValue: 1.3, decay: 2, dropRate: 2, limit: 0);

      expect(
          elements, iterableAlmostEqualTo([], 1e-2));
    });
  });
}
