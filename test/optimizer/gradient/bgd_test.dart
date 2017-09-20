import 'dart:typed_data';
import 'package:di/di.dart';
import 'package:test/test.dart';
import 'package:mockito/mockito.dart';
import 'package:simd_vector/vector.dart';
import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/interface.dart';
import 'package:dart_ml/src/implementation.dart';
import 'package:dart_ml/src/loss_function/loss_function.dart';
import 'package:dart_ml/src/score_function/score_function.dart';

class LearningRateGeneratorMock extends Mock implements LearningRateGenerator {
  void init(double initialValue) {}
  double generate(int iterationNumber) => 1.0;
}

class GradientCalculatorMock extends Mock implements GradientCalculator {
  int _counter = 0;

  void init(int numberOfArguments, double argumentDelta, TargetFunction function) {}

  Float32x4Vector getGradient(Float32x4Vector k, Float32x4Vector x, double y) {
    _counter++;

    switch(_counter) {
      case 1:
        return new Float32x4Vector.from([1.1, 2.1, 3.1]);
      case 2:
        return new Float32x4Vector.from([2.1, 3.1, 4.1]); //-1.6, -2.6, -3.6
      case 3:
        return new Float32x4Vector.from([3.1, 4.1, 5.1]);
      case 4:
        return new Float32x4Vector.from([4.1, 5.1, 6.1]); //[-1.6, -2.6, -3.6] - [3.6, 4.6, 5.6] = [-5.2, -7.2, -9.2]
      default:
        throw new Error();
    }
  }
}

void main() {
  group('Batch gradient descent optimizer', () {
    BGDOptimizer optimizer;
    List<Float32x4Vector> data;
    Float32List target;

    setUp(() {
      injector = new ModuleInjector([
        new Module()
          ..bind(LearningRateGenerator, toFactory: () => new LearningRateGeneratorMock())
          ..bind(GradientCalculator, toFactory: () => new GradientCalculatorMock())
      ]);

      optimizer = GradientOptimizerFactory.createBatchOptimizer();

      optimizer.configure(
        learningRate: 1e-5,
        minWeightsDistance: null,
        iterationLimit: 2,
        regularization: Regularization.L2,
        alpha: .00001,
        lossFunction: new LossFunction.Squared(),
        scoreFunction: new ScoreFunction.Linear()
      );

      data = [
        new Float32x4Vector.from([5.0, 10.0, 15.0]),
        new Float32x4Vector.from([1.0, 2.0, 3.0])
      ];

      target = new Float32List.fromList([10.0, 20.0]);
    });

    test('should find optimal weights for the given data', () {
      Float32x4Vector weights = optimizer.findMinima(data, target);
      List<double> formattedWeights = weights.asList().map((double value) => double.parse(value.toStringAsFixed(2)))
          .toList();

      expect(formattedWeights, [-5.2, -7.2, -9.2]);
    });
  });
}
