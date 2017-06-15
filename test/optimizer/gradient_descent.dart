import 'package:di/di.dart';
import 'package:test/test.dart';
import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/math/math.dart' show Randomizer;
import 'package:dart_ml/src/math/math_impl.dart' show Vector, RandomizerImpl;
import 'package:dart_ml/src/optimizer/optimizer.dart' show Regularization;
import 'package:dart_ml/src/optimizer/optimizer_impl.dart' show BGDOptimizerImpl, MBGDOptimizerImpl, SGDOptimizerImpl;
import 'package:dart_ml/src/loss_function/squared_loss.dart';

void main() {
  const int MAX_EPOCH = 200;
  List<Vector> data;
  Vector target;

  setUp(() {
    injector = new ModuleInjector([
      new Module()
        ..bind(Randomizer, toFactory: () => new RandomizerImpl())
    ]);

    data = [
      new Vector.from([230.1, 37.8, 69.2]),
      new Vector.from([44.5, 39.3, 45.1]),
      new Vector.from([17.2, 45.9, 69.3])
    ];

    target = new Vector.from([22.1, 10.4, 9.3]);
  });

  group('Batch gradient descent optimizer', () {
    BGDOptimizerImpl optimizer;

    setUp(() {
      optimizer = new BGDOptimizerImpl();
      optimizer.configure(1e-5, 1e-8, 100, Regularization.L2, new SquaredLoss(), alpha: .00001);
    });

    test('should find optimal weights for the given data', () {
      Vector weightsSum = new Vector.zero(3);

      for (int i = 0; i < MAX_EPOCH; i++) {
        weightsSum += optimizer.optimize(data, target);
      }

      Vector weightsMean = weightsSum.scalarMul(1 / MAX_EPOCH);

      expect(weightsMean.length, equals(3));
      expect(0.07 < weightsMean[0] && weightsMean[0] < 0.08, isTrue, reason: '${weightsMean[0]} not in interval (0.07...0.08)');
      expect(0.02 < weightsMean[1] && weightsMean[1] < 0.03, isTrue, reason: '${weightsMean[1]} not in interval (0.02...0.03)');
      expect(0.04 < weightsMean[2] && weightsMean[2] < 0.05, isTrue, reason: '${weightsMean[2]} not in interval (0.04...0.05)');
    });
  });

  group('Mini batch gradient descent optimizer', () {
    MBGDOptimizerImpl optimizer;

    setUp(() {
      optimizer = new MBGDOptimizerImpl();
      optimizer.configure(1e-5, 1e-8, 100, Regularization.L2, new SquaredLoss(), alpha: .00001);
    });

    test('should find optimal weights for the given data', () {
      Vector weights = optimizer.optimize(data, target);
      expect(weights.length, equals(3));
      expect(0.07 < weights[0] && weights[0] < 0.09, isTrue, reason: '${weights[0]} not in interval (0.08...0.09)');
      expect(0.01 < weights[1] && weights[1] < 0.03, isTrue, reason: '${weights[1]} not in interval (0.01...0.03)');
      expect(0.02 < weights[2] && weights[2] < 0.04, isTrue, reason: '${weights[2]} not in interval (0.03...0.04)');
    });
  });

  group('Stochastic gradient descent optimizer', () {
    SGDOptimizerImpl optimizer;

    setUp(() {
      optimizer = new SGDOptimizerImpl();
      optimizer.configure(1e-5, 1e-8, 100, Regularization.L2, new SquaredLoss(), alpha: .00001);
    });

    test('should find optimal weights for the given data', () {
      Vector weightsSum = new Vector.zero(3);

      for (int i = 0; i < MAX_EPOCH; i++) {
        weightsSum += optimizer.optimize(data, target);
      }

      Vector weightsMean = weightsSum.scalarMul(1 / MAX_EPOCH);

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
