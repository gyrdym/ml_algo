import 'dart:typed_data';
import 'package:di/di.dart';
import 'package:test/test.dart';
import 'package:mockito/mockito.dart';
import 'package:simd_vector/vector.dart';
import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/interface.dart';
import 'package:dart_ml/src/implementation.dart';

///Randomizer for mini batch gradient descent.
class MBGDRandomizerMock extends Mock implements Randomizer {
  int _iterationCounter = 0;

  List<int> getIntegerInterval(int start, int end) {
    List<int> interval;

    if (_iterationCounter % 2 == 0) {
      interval = [1, 2];
    } else {
      interval = [0, 1];
    }

    _iterationCounter++;
    return interval;
  }
}

void main() {
  List<Float32x4Vector> data;
  Float32List target;

  setUp(() {
    data = [
      new Float32x4Vector.from([230.1, 37.8, 69.2]),
      new Float32x4Vector.from([44.5, 39.3, 45.1]),
      new Float32x4Vector.from([17.2, 45.9, 69.3])
    ];

    target = new Float32List.fromList([22.1, 10.4, 9.3]);
  });

  group('Mini batch gradient descent optimizer', () {
    MBGDOptimizer optimizer;

    setUp(() {
      injector = new ModuleInjector([
        new Module()
          ..bind(Randomizer, toFactory: () => new MBGDRandomizerMock())
      ]);

      optimizer = GradientOptimizerFactory.createMiniBatchOptimizer();
      optimizer.configure(
        learningRate: 1e-5,
        minWeightsDistance: null,
        iterationLimit:  10,
        regularization: Regularization.L2,
        alpha: .00001
      );
    });

    test('should find optimal weights for the given data', () {
      Float32x4Vector weights = optimizer.findMinima(data, target);

      expect(weights.asList(), equals([]));
    });
  });
}
