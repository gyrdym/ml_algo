import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_ml/dart_ml.dart';
import 'package:dart_ml/src/math/math_analysis/gradient_calculator.dart';
import 'package:dart_ml/src/math/math_analysis/gradient_calculator_impl.dart';
import 'package:matcher/matcher.dart';
import 'package:test/test.dart';

void main() {
  GradientCalculator<Float32x4> calculator;

  group('GradientCalculator', () {
    setUp(() {
      calculator = GradientCalculatorImpl();
    });

    test('should return a proper gradient vector', () {
      final k = Float32x4VectorFactory.from([0.3, 0.7, 0.4]);
      final x = Float32x4VectorFactory.from([1.0, 2.0, 3.0]);
      final y = 1.0;

      final gradient = calculator.getGradient(
        (
          MLVector<Float32x4> a,
          Iterable<MLVector<Float32x4>> vectorArgs,
          Iterable<double> scalarArgs
        ) {
          final b = (vectorArgs as List<MLVector<Float32x4>>)[0];
          final c = (scalarArgs as List<double>)[0];
          return math.pow(a.dot(b) - c, 2).toDouble();
        }, k, [x], [y], 0.00001)
      .toList()
      .map((double value) => double.parse(value.toStringAsFixed(2))).toList();

      expect(gradient, equals([3.81, 7.61, 11.42]));
    });
  });
}