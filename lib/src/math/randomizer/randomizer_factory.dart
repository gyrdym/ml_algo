import 'package:dart_ml/src/math/randomizer/randomizer.dart';
import 'package:dart_ml/src/math/randomizer/randomizer_impl.dart';

class RandomizerFactory {
  static Randomizer Default(int seed) => new RandomizerImpl(seed: seed);
}