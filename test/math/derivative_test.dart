import 'dart:math' as math;
import 'package:dart_ml/dart_ml.dart';
import 'package:dart_ml/src/interface.dart';
import 'package:dart_ml/src/implementation.dart';
import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

void main() {
  DerivativeFinder finder;

  group('Derivative finder', () {
    setUp(() {
      finder = MathUtils.createDerivativeFinder();
      finder.configure(3, 0.00001, (Float32x4Vector a, Float32x4Vector b, double c) {
        return math.pow(a.dot(b) - c, 2);
      });
    });

    test('should return an approximately value of the derivative', () {
      finder.partialDerivative(new Float32x4Vector.from([]), new Float32x4Vector.from([]), 2.0, 0);
//      expect(start <= value && value < end, isTrue);
    });
  });
}