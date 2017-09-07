import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:simd_vector/vector.dart';
import 'package:dart_ml/src/interface.dart';
import 'package:dart_ml/src/implementation.dart';
import 'package:dart_ml/src/loss_function/loss_function.dart';
import 'package:dart_ml/src/score_function/score_function.dart';

void main() {
  group('Batch gradient descent optimizer', () {
    BGDOptimizer optimizer;
    List<Float32x4Vector> data;
    Float32List target;

    setUp(() {
      optimizer = GradientOptimizerFactory.createBatchOptimizer();

      optimizer.configure(
        learningRate: 1e-5,
        minWeightsDistance: null,
        iterationLimit:  10,
        regularization: Regularization.L2,
        alpha: .00001,
        lossFunction: new LossFunction.Squared(),
        scoreFunction: new ScoreFunction.Linear()
      );

      data = [
        new Float32x4Vector.from([230.1, 37.8, 69.2]),
        new Float32x4Vector.from([44.5, 39.3, 45.1])
      ];

      target = new Float32List.fromList([22.1, 10.4]);
    });

    test('should find optimal weights for the given data', () {
      Float32x4Vector weights = optimizer.findMinima(data, target);
      print(weights);
      expect(weights.asList(), []);
    });
  });
}
