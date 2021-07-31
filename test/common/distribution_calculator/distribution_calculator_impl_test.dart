import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator_impl.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() {
  group('DistributionCalculatorImpl', () {
    group('when a collection of numerical values is used', () {
      test(
          'should calculate probability distribution of the given '
          'sequence', () {
        final labels = [1, 1, 1, 2, 3];
        final calculator = const DistributionCalculatorImpl();
        final actual = calculator.calculate(labels);
        final expected = {
          1: closeTo(3 / 5, 1e-4),
          2: 1 / 5,
          3: 1 / 5,
        };
        expect(actual, equals(expected));
      });

      test(
          'should return a map with one entry where the value (probability) '
          'equal to for the given sequence consisting of one repeating '
          'value', () {
        final values = [1, 1, 1, 1, 1];
        final calculator = const DistributionCalculatorImpl();
        final actual = calculator.calculate(values);
        final expected = {
          1: 1.0,
        };
        expect(actual, equals(expected));
      });

      test(
          'should return a map where all values (probabilities) are uniform '
          'if the given sequence elements are all different', () {
        final values = [10, 20, 30, 40, 60];
        final calculator = const DistributionCalculatorImpl();
        final actual = calculator.calculate(values);
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

    group('when a collection of vectors is used', () {
      test(
          'should calculate probability distribution of the given '
          'sequence', () {
        final values = [
          Vector.fromList([1, 0, 0]),
          Vector.fromList([0, 0, 1]),
          Vector.fromList([0, 0, 1]),
          Vector.fromList([0, 1, 0]),
          Vector.fromList([1, 0, 0]),
        ];
        final calculator = const DistributionCalculatorImpl();
        final actual = calculator.calculate(values);
        final expected = {
          Vector.fromList([1, 0, 0]): 2 / 5,
          Vector.fromList([0, 0, 1]): 2 / 5,
          Vector.fromList([0, 1, 0]): 1 / 5,
        };
        expect(actual, equals(expected));
      });

      test(
          'should return a map with one entry where the value (probability) '
          'equal to 1 if the given sequence consists just of one repeating '
          'vector', () {
        final values = [
          Vector.fromList([1, 0, 0]),
          Vector.fromList([1, 0, 0]),
          Vector.fromList([1, 0, 0]),
          Vector.fromList([1, 0, 0]),
        ];
        final calculator = const DistributionCalculatorImpl();
        final actual = calculator.calculate(values);
        final expected = {
          Vector.fromList([1, 0, 0]): 1.0,
        };
        expect(actual, equals(expected));
      });

      test(
          'should return a map where all values (probabilities) are uniform '
          'if the given sequence elements are all different', () {
        final values = [
          Vector.fromList([1, 0, 0]),
          Vector.fromList([1, 2, 3]),
          Vector.fromList([4, 5, 6]),
        ];
        final calculator = const DistributionCalculatorImpl();
        final actual = calculator.calculate(values);
        final expected = {
          Vector.fromList([1, 0, 0]): closeTo(1 / 3, 1e-4),
          Vector.fromList([1, 2, 3]): closeTo(1 / 3, 1e-4),
          Vector.fromList([4, 5, 6]): closeTo(1 / 3, 1e-4),
        };
        expect(actual, equals(expected));
      });
    });

    test(
        'should use length from arguments if the appropriate argument is '
        'provided', () {
      final values = [
        'class 1',
        'class 2',
        'class 3',
      ];
      final calculator = const DistributionCalculatorImpl();
      final actual = calculator.calculate(values, 10);
      final expected = {
        'class 1': 1 / 10,
        'class 2': 1 / 10,
        'class 3': 1 / 10,
      };
      expect(actual, equals(expected));
    });

    test(
        'should throw an error if value collection is empty (provided length '
        'should be ignored)', () {
      final values = <double>[];
      final calculator = const DistributionCalculatorImpl();
      final actual = () => calculator.calculate(values, 10);

      expect(actual, throwsException);
    });

    test('should throw an error if given length is 0', () {
      final values = <double>[1, 2, 3];
      final calculator = const DistributionCalculatorImpl();
      final actual = () => calculator.calculate(values, 0);

      expect(actual, throwsException);
    });
  });
}
