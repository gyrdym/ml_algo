import 'package:dart_ml/src/cost_function/cost_function_factory.dart';
import 'package:simd_vector/vector.dart';
import 'package:test/test.dart';

void main() {
  group('Log-likelihood cost function', () {
    final logLikelihood = CostFunctionFactory.LogLikelihood();

    test('should calculate correct value of the partial derivative (in case of True-label)', () {
      final x = new Float32x4Vector.from([3.0, 1.2, 5.6]);
      final w = new Float32x4Vector.from([1.5, 2.3, 3.2]);
      final precision = 0.00001;
      final y = 1.0;

      // partial derivative with respect to i-member = x_i * (indicator(y=1) - probability(y=1|x,w))
      // probability(y=1|x,w) = 1 / (1 + e^(-score))
      // score = x.w (dot product)
      // so, common formula for this test: x_i * (1 - (1 / (1 + e^(-1 * [3.0, 1.2, 5.6].[1.5, 2.3, 3.2])))) =
      // x_i * (1 - (1 / (1 + e^-25.08))) = x_i * 1.2820189354556533e-11

      expect(logLikelihood.getPartialDerivative(0, x, w, y), closeTo(3.8e-11, precision));
      expect(logLikelihood.getPartialDerivative(1, x, w, y), closeTo(1.5e-11, precision));
      expect(logLikelihood.getPartialDerivative(2, x, w, y), closeTo(7.1e-11, precision));
    });

    test('should calculate correct value of the partial derivative (in case of False-label)', () {
      final x = new Float32x4Vector.from([3.0, 1.2, 5.6]);
      final w = new Float32x4Vector.from([1.5, 2.3, 3.2]);
      final precision = 0.00001;
      final y = 0.0;

      // partial derivative with respect to i-member = x_i * (indicator(y=1) - probability(y=1|x,w))
      // probability(y=1|x,w) = 1 / (1 + e^(-score))
      // score = x.w (dot product)
      // so, common formula for this test: x_i * (0 - (1 / (1 + e^(-1 * [3.0, 1.2, 5.6].[1.5, 2.3, 3.2])))) =
      // x_i * (0 - (1 / (1 + e^-25.08))) = x_i * -0.9999999999871798

      expect(logLikelihood.getPartialDerivative(0, x, w, y), closeTo(-3.0, precision));
      expect(logLikelihood.getPartialDerivative(1, x, w, y), closeTo(-1.2, precision));
      expect(logLikelihood.getPartialDerivative(2, x, w, y), closeTo(-5.6, precision));
    });
  });
}