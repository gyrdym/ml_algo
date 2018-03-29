import 'dart:math' as math;

import 'package:dart_ml/dart_ml.dart';
import 'package:dart_ml/src/core/implementation.dart';
import 'package:dart_ml/src/core/interface.dart';
import 'package:matcher/matcher.dart';
import 'package:test/test.dart';

void main() {
  GradientCalculator calculator;

  group('GradientCalculator', () {
    setUp(() {
      calculator = MathUtils.createGradientCalculator();
    });

    test('should return a proper gradient vector', () {
      final k = new Float32x4Vector.from([0.3, 0.7, 0.4]);
      final x = new Float32x4Vector.from([1.0, 2.0, 3.0]);
      final y = 1.0;

      final gradient = calculator.getGradient(
        (Float32x4Vector a, List<Float32x4Vector> vectorArgs, List<double> scalarArgs) {
          final b = vectorArgs[0];
          final c = scalarArgs[0];
          return math.pow(a.dot(b) - c, 2);
        }, k, [x], [y], 0.00001
      )
      .asList()
      .map((double value) => double.parse(value.toStringAsFixed(2))).toList();

      expect(gradient, equals([3.81, 7.61, 11.42]));
    });
  });
}