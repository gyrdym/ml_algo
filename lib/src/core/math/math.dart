import 'package:dart_ml/src/core/math/math_analysis/gradient_calculator.dart';
import 'package:dart_ml/src/core/math/math_analysis/gradient_calculator_impl.dart';
import 'package:dart_ml/src/core/math/randomizer/randomizer.dart';
import 'package:dart_ml/src/core/math/randomizer/randomizer_impl.dart';

class MathUtils {
  static Randomizer createRandomizer({int seed}) => new RandomizerImpl(seed: seed);
  static GradientCalculator createGradientCalculator() => new GradientCalculatorImpl();
}