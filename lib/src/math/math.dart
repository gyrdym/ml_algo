part of 'package:dart_ml/src/implementation.dart';

class MathUtils {
  static Randomizer createRandomizer({int seed}) => new _RandomizerImpl(seed: seed);
}