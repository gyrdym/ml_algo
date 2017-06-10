import 'package:di/di.dart';
import 'package:test/test.dart';
import 'package:mockito/mockito.dart';
import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/math/math.dart' show Randomizer;
import 'package:dart_ml/src/math/math_impl.dart' show Vector;
import 'package:dart_ml/src/optimizer/optimizer.dart' show Regularization;
import 'package:dart_ml/src/optimizer/optimizer_impl.dart' show BGDOptimizerImpl, MBGDOptimizerImpl, SGDOptimizerImpl;

class RandomizerMock extends Mock implements Randomizer {}

void main() {
  List<Vector> data;
  Vector target;

  setUp(() {
    injector = new ModuleInjector([
      new Module()
        ..bind(Randomizer, toFactory: () => new RandomizerMock())
    ]);

    data = [
      new Vector.from([230.1, 37.8, 69.2]),
      new Vector.from([44.5, 39.3, 45.1]),
      new Vector.from([17.2, 45.9, 69.3]),
      new Vector.from([151.5, 41.3, 58.5]),
      new Vector.from([180.8, 10.8, 58.4]),
      new Vector.from([8.7, 48.9, 75.0]),
      new Vector.from([57.5, 32.8, 23.5]),
      new Vector.from([120.2, 19.6, 11.6]),
      new Vector.from([8.6, 2.1, 1.0]),
      new Vector.from([199.8, 2.6, 21.2])
    ];

    target = new Vector.from([22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 4.8, 10.6]);
  });

  group('Batch gradient descent ', () {
    BGDOptimizerImpl optimizer;

    setUp(() {
      optimizer = new BGDOptimizerImpl();
      optimizer.configure(1e-5, 1e-8, 10, Regularization.L2);
    });

    test('should find optimal weights for the given data', () {
      expect(optimizer.optimize(data, target), equals([]));
    });
  });
}
