import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_impl.dart';

class RandomizerFactory {
  static Randomizer defaultRandomizer(int seed) => RandomizerImpl(seed: seed);
}
