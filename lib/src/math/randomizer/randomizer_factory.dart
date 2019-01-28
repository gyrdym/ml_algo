import 'package:ml_algo/src/math/randomizer/randomizer.dart';

abstract class RandomizerFactory {
  Randomizer create([int seed]);
}
