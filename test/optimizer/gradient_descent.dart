import 'package:di/di.dart';
import 'package:test/test.dart';
import 'package:simd_vector/vector.dart';
import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/interface.dart';
import 'package:dart_ml/src/implementation.dart';
import 'package:dart_ml/src/math/math.dart' show Randomizer;
import 'package:dart_ml/src/math/math_impl.dart' show RandomizerImpl;
import 'dart:typed_data' show Float32List;

void main() {
  const int MAX_EPOCH = 200;
  List<Float32x4Vector> data;
  List<double> target;

  setUp(() {
    injector = new ModuleInjector([
      new Module()
        ..bind(Randomizer, toFactory: () => new RandomizerImpl())
    ]);

    data = [
      new Float32x4Vector.from([230.1, 37.8, 69.2]),
      new Float32x4Vector.from([44.5, 39.3, 45.1]),
      new Float32x4Vector.from([17.2, 45.9, 69.3])
    ];

    target = [22.1, 10.4, 9.3];
  });

  group('Batch gradient descent optimizer', () {
    BGDOptimizer optimizer;

    setUp(() {
      optimizer = GradientOptimizerFactory.createBatchOptimizer();
      optimizer.configure(
        learningRate: 1e-5,
        minWeightsDistance: 1e-8,
        iterationLimit:  100,
        regularization: Regularization.L2,
        alpha: .00001
      );
    });

    test('should find optimal weights for the given data', () {
      Float32x4Vector weightsSum = new Float32x4Vector.zero(3);

      for (int i = 0; i < MAX_EPOCH; i++) {
        weightsSum += optimizer.findMinima(data, target);
      }

      Float32List weightsMean = weightsSum.scalarMul(1 / MAX_EPOCH).asList();

      expect(weightsMean.length, equals(3));
      expect(0.07 < weightsMean[0] && weightsMean[0] < 0.08, isTrue, reason: '${weightsMean[0]} not in interval (0.07...0.08)');
      expect(0.02 < weightsMean[1] && weightsMean[1] < 0.03, isTrue, reason: '${weightsMean[1]} not in interval (0.02...0.03)');
      expect(0.04 < weightsMean[2] && weightsMean[2] < 0.05, isTrue, reason: '${weightsMean[2]} not in interval (0.04...0.05)');
    });
  });

  group('Mini batch gradient descent optimizer', () {
    MBGDOptimizer optimizer;

    setUp(() {
      optimizer = GradientOptimizerFactory.createMiniBatchOptimizer();
      optimizer.configure(
        learningRate: 1e-5,
        minWeightsDistance: 1e-8,
        iterationLimit:  100,
        regularization: Regularization.L2,
        alpha: .00001
      );
    });

    test('should find optimal weights for the given data', () {
      Float32List weights = optimizer.findMinima(data, target).asList();

      expect(weights.length, equals(3));
      expect(0.07 < weights[0] && weights[0] < 0.09, isTrue, reason: '${weights[0]} not in interval (0.08...0.09)');
      expect(0.01 < weights[1] && weights[1] < 0.03, isTrue, reason: '${weights[1]} not in interval (0.01...0.03)');
      expect(0.02 < weights[2] && weights[2] < 0.04, isTrue, reason: '${weights[2]} not in interval (0.03...0.04)');
    });
  });

  group('Stochastic gradient descent optimizer', () {
    SGDOptimizer optimizer;

    setUp(() {
      optimizer = GradientOptimizerFactory.createStochasticOptimizer();
      optimizer.configure(
        learningRate: 1e-5,
        minWeightsDistance: 1e-8,
        iterationLimit:  100,
        regularization: Regularization.L2,
        alpha: .00001
      );
    });

    test('should find optimal weights for the given data', () {
      Float32x4Vector weightsSum = new Float32x4Vector.zero(3);

      for (int i = 0; i < MAX_EPOCH; i++) {
        weightsSum += optimizer.findMinima(data, target);
      }

      Float32List weightsMean = weightsSum.scalarMul(1 / MAX_EPOCH).asList();

      expect(weightsMean.length, equals(3));
      expect(0.07 < weightsMean[0] && weightsMean[0] < 0.08, isTrue,
                 reason: '${weightsMean[0]} not in interval (0.08...0.09)');
      expect(0.02 < weightsMean[1] && weightsMean[1] < 0.03, isTrue,
                 reason: '${weightsMean[1]} not in interval (0.01...0.03)');
      expect(0.04 < weightsMean[2] && weightsMean[2] < 0.045, isTrue,
                 reason: '${weightsMean[2]} not in interval (0.03...0.04)');
    });
  });
}
