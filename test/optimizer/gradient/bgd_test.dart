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

class GradientCalculatorMock extends Mock implements GradientCalculator {
  int _counter = 0;

  void init(int numberOfArguments, double argumentDelta, TargetFunction function) {}

  Float32x4Vector getGradient(Float32x4Vector k, Float32x4Vector x, double y) {
    _counter++;

    switch(_counter) {
      case 1:
        return new Float32x4Vector.from([1.0, 2.0, 3.0]);
      case 2:
        return new Float32x4Vector.from([2.0, 3.0, 4.0]); //1.5 2.5 3.5 - -1.5 -2.5 -3.5
      case 3:
        return new Float32x4Vector.from([3.0, 4.0, 5.0]);
      case 4:
        return new Float32x4Vector.from([4.0, 5.0, 6.0]); //3.5 4.5 5.5 - -5.0 -7.0 -9.0
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
      expect(weights.asList(), [-5.0, -7.0, -9.0]);
    });
  });
}
