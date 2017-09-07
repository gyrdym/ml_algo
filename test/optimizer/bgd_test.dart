import 'package:test/test.dart';
import 'package:simd_vector/vector.dart';
import 'package:dart_ml/src/interface.dart';
import 'package:dart_ml/src/implementation.dart';

void main() {
  group('Batch gradient descent optimizer', () {
    BGDOptimizer optimizer;
    List<Float32x4Vector> data;
    List<double> target;

    setUp(() {
      optimizer = GradientOptimizerFactory.createBatchOptimizer();

      optimizer.configure(
        learningRate: 1e-5,
        minWeightsDistance: 1e-8,
        iterationLimit:  10,
        regularization: Regularization.L2,
        alpha: .00001
      );

      data = [
        new Float32x4Vector.from([230.1, 37.8, 69.2]),
        new Float32x4Vector.from([44.5, 39.3, 45.1])
      ];

      target = [22.1, 10.4, 9.3];
    });

    test('should find optimal weights for the given data', () {
      Float32x4Vector weights = optimizer.findMinima(data, target);
      expect(weights.asList(), []);
    });
  });
}
