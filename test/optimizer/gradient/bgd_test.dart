import 'dart:typed_data';
import 'package:di/di.dart';
import 'package:test/test.dart';
import 'package:mockito/mockito.dart';
import 'package:simd_vector/vector.dart';
import 'package:dart_ml/src/interface.dart';
import 'package:dart_ml/src/implementation.dart';

const int ITERATIONS_NUMBER = 2;

class InitialWeightsGeneratorMock extends Mock implements InitialWeightsGenerator {}
class LearningRateGeneratorMock extends Mock implements LearningRateGenerator {}

class GradientCalculatorMock extends Mock implements GradientCalculator {
  int _counter = 0;

  Float32x4Vector getGradient(Float32x4Vector k, Float32x4Vector x, double y) {
    switch(_counter++) {
      case 0:
        return new Float32x4Vector.from([1.1, 2.1, 3.1]);
      case 1:
        return new Float32x4Vector.from([2.1, 3.1, 4.1]);
      case 2:
        return new Float32x4Vector.from([3.1, 4.1, 5.1]);
      case 3:
        return new Float32x4Vector.from([4.1, 5.1, 6.1]);
      default:
        throw new Error();
    }
  }
}

void main() {
  group('Batch gradient descent optimizer', () {
    LearningRateGeneratorMock learningRateGeneratorMock;
    GradientCalculatorMock gradientCalculatorMock;
    InitialWeightsGeneratorMock weightsGenerator;

    BGDOptimizer optimizer;
    List<Float32x4Vector> data;
    Float32List target;

    setUp(() {
      learningRateGeneratorMock = new LearningRateGeneratorMock();
      gradientCalculatorMock = new GradientCalculatorMock();
      weightsGenerator = new InitialWeightsGeneratorMock();

      injector = new ModuleInjector([
        new Module()
          ..bind(LearningRateGenerator, toValue: learningRateGeneratorMock)
          ..bind(GradientCalculator, toValue: gradientCalculatorMock)
          ..bind(InitialWeightsGenerator, toValue: weightsGenerator)
      ]);

      optimizer = GradientOptimizerFactory.createBatchOptimizer(1e-5, null, ITERATIONS_NUMBER, Regularization.L2, .00001, .0001);

      data = [
        new Float32x4Vector.from([5.0, 10.0, 15.0]),
        new Float32x4Vector.from([1.0, 2.0, 3.0])
      ];

      target = new Float32List.fromList([10.0, 20.0]);

      when(learningRateGeneratorMock.getNextValue()).thenReturn(1.0);
      when(weightsGenerator.generate(3)).thenReturn(new Float32x4Vector.from([0.0, 0.0, 0.0]));
    });

    tearDown(() {
      verify(learningRateGeneratorMock.getNextValue()).called(ITERATIONS_NUMBER);
    });

    test('should find optimal weights for the given data', () {
      Float32x4Vector weights = optimizer.findMinima(data, target);
      List<double> formattedWeights = weights.asList().map((double value) => double.parse(value.toStringAsFixed(2)))
          .toList();
      expect(formattedWeights, equals([-5.2, -7.2, -9.2]));
    });
  });
}
