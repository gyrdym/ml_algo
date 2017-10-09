import 'dart:math' as math;
import 'package:dart_ml/dart_ml.dart';
import 'package:dart_ml/src/core/interface.dart';
import 'package:dart_ml/src/core/implementation.dart';
import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

void main() {
  GradientCalculator calculator;

  group('Derivative finder', () {
    setUp(() {
      calculator = MathUtils.createGradientCalculator();
      calculator.init(3, 0.00001, (Float32x4Vector a, Float32x4Vector b, double c) {
        return math.pow(a.dot(b) - c, 2);
      });
    });

    test('should return a proper gradient vector', () {
      Float32x4Vector k = new Float32x4Vector.from([0.3, 0.7, 0.4]);
      Float32x4Vector x = new Float32x4Vector.from([1.0, 2.0, 3.0]);
      double y = 1.0;

      List<double> gradient = calculator.getGradient(k, x, y).asList()
          .map((double value) => double.parse(value.toStringAsFixed(2))).toList();

      expect(gradient, equals([3.81, 7.61, 11.42]));
    });
  });
}