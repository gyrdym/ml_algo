import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/iterables/constants.dart';
import 'package:test/test.dart';

void main() {
  group('ConstantLearningRateIterable', () {
    test('should return a sequence of the same elements, iterationLimit=5', () {
      const iterationLimit = 5;
      final elements = ConstantLearningRateIterable(1.3, iterationLimit);

      expect(elements, [1.3, 1.3, 1.3, 1.3, 1.3]);
    });

    test('should return a sequence of the same elements, iterationLimit=1', () {
      const iterationLimit = 1;
      final elements = ConstantLearningRateIterable(3.5, iterationLimit);

      expect(elements, [3.5]);
    });

    test('should return a sequence of the same elements, iterationLimit=0', () {
      const iterationLimit = 0;
      final elements = ConstantLearningRateIterable(1.3, iterationLimit);

      expect(elements, []);
    });
  });
}
