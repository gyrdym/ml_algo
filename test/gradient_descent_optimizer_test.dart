import 'package:di/di.dart';
import 'package:test/test.dart';
import 'package:mockito/mockito.dart';
import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/math/math.dart' show Randomizer;
import 'package:dart_ml/src/math/math_impl.dart' show Vector;
import 'package:dart_ml/src/optimizer/optimizer_impl.dart' show BGDOptimizerImpl, MBGDOptimizerImpl, SGDOptimizerImpl;

class RandomizerMock extends Mock implements Randomizer {}

void main() {
  List<Vector> data = [
    new Vector.from([230.1, 37.8, 69.2]),
    new Vector.from([]),
    new Vector.from([]),
    new Vector.from([]),
    new Vector.from([]),
    new Vector.from([]),
    new Vector.from([]),
    new Vector.from([]),
    new Vector.from([]),
    new Vector.from([])
  ];

  injector = new ModuleInjector([
    new Module()
      ..bind(Randomizer, toFactory: () => new RandomizerMock())
  ]);
}
