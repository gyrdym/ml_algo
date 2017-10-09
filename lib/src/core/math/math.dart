part of 'package:dart_ml/src/core/implementation.dart';

class MathUtils {
  static Randomizer createRandomizer({int seed}) => new _RandomizerImpl(seed: seed);
  static GradientCalculator createGradientCalculator() => new _GradientCalculatorImpl();
}