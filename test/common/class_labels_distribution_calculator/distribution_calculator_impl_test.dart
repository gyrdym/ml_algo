import 'package:ml_algo/src/common/class_labels_distribution_calculator/distribution_calculator_impl.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() {
  group('ClassLabelsDistributionCalculatorImpl', () {
    group('when a numerical value used as a class label', () {
      test('should calculate probability distribution of the given sequence of '
          'class labels', () {
        final classLabels = [1, 1, 1, 2, 3];
        final calculator = ClassLabelsDistributionCalculatorImpl();
        final actual = calculator.calculate(classLabels, 5);
        final expected = {
          1: closeTo(3 / 5, 1e-4),
          2: 1 / 5,
          3: 1 / 5,
        };
        expect(actual, equals(expected));
      });

      test('should return a map with one entry where the value (probability) '
          'equals to 1 if the given class label sequence consists just of one '
          'class label', () {
        final classLabels = [1, 1, 1, 1, 1];
        final calculator = ClassLabelsDistributionCalculatorImpl();
        final actual = calculator.calculate(classLabels, 5);
        final expected = {
          1: 1.0,
        };
        expect(actual, equals(expected));
      });

      test('should return a map where all values (probabilities) are uniform '
          'if the given class label sequence values are all different', () {
        final classLabels = [10, 20, 30, 40, 60];
        final calculator = ClassLabelsDistributionCalculatorImpl();
        final actual = calculator.calculate(classLabels, 5);
        final expected = {
          10: 1 / 5,
          20: 1 / 5,
          30: 1 / 5,
          40: 1 / 5,
          60: 1 / 5,
        };
        expect(actual, equals(expected));
      });
    });

    group('when a vector used as a class label', () {
      test('should calculate probability distribution of the given sequence of '
          'class labels', () {
        final classLabels = [
          Vector.fromList([1, 0, 0]),
          Vector.fromList([0, 0, 1]),
          Vector.fromList([0, 0, 1]),
          Vector.fromList([0, 1, 0]),
          Vector.fromList([1, 0, 0]),
        ];
        final calculator = ClassLabelsDistributionCalculatorImpl();
        final actual = calculator.calculate(classLabels, 5);
        final expected = {
          Vector.fromList([1, 0, 0]): 2 / 5,
          Vector.fromList([0, 0, 1]): 2 / 5,
          Vector.fromList([0, 1, 0]): 1 / 5,
        };
        expect(actual, equals(expected));
      });

      test('should return a map with one entry where the value (probability) '
          'equals to 1 if the given class label sequence consists just of one '
          'class label', () {
        final classLabels = [
          Vector.fromList([1, 0, 0]),
          Vector.fromList([1, 0, 0]),
          Vector.fromList([1, 0, 0]),
          Vector.fromList([1, 0, 0]),
        ];
        final calculator = ClassLabelsDistributionCalculatorImpl();
        final actual = calculator.calculate(classLabels, 4);
        final expected = {
          Vector.fromList([1, 0, 0]): 1.0,
        };
        expect(actual, equals(expected));
      });

      test('should return a map where all values (probabilities) are uniform '
          'if the given class label sequence values are all different', () {
        final classLabels = [
          Vector.fromList([1, 0, 0]),
          Vector.fromList([1, 2, 3]),
          Vector.fromList([4, 5, 6]),
        ];
        final calculator = ClassLabelsDistributionCalculatorImpl();
        final actual = calculator.calculate(classLabels, 3);
        final expected = {
          Vector.fromList([1, 0, 0]): closeTo(1 / 3, 1e-4),
          Vector.fromList([1, 2, 3]): closeTo(1 / 3, 1e-4),
          Vector.fromList([4, 5, 6]): closeTo(1 / 3, 1e-4),
        };
        expect(actual, equals(expected));
      });
    });
  });
}
