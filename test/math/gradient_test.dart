import 'dart:math' as math;

import 'package:dart_ml/dart_ml.dart';
import 'package:dart_ml/src/math/math_analysis/gradient_calculator.dart';
import 'package:dart_ml/src/math/math_analysis/gradient_calculator_impl.dart';
import 'package:matcher/matcher.dart';
import 'package:test/test.dart';

void main() {
  GradientCalculator calculator;

  group('GradientCalculator', () {
    setUp(() {
      calculator = new GradientCalculatorImpl();
    });

    test('should return a proper gradient vector', () {
      final k = new Float32x4Vector.from([0.3, 0.7, 0.4]);
      final x = new Float32x4Vector.from([1.0, 2.0, 3.0]);
      final y = 1.0;

      final gradient = calculator.getGradient(
        (Vector a, Iterable<Vector> vectorArgs, Iterable<double> scalarArgs) {
          final b = (vectorArgs as List)[0];
          final c = (scalarArgs as List)[0];
          return math.pow(a.dot(b) - c, 2);
        }, k, [x], [y], 0.00001
      )
      .asList()
      .map((double value) => double.parse(value.toStringAsFixed(2))).toList();

      expect(gradient, equals([3.81, 7.61, 11.42]));
    });
  });
}