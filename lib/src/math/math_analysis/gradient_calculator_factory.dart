import 'package:ml_algo/src/math/math_analysis/gradient_calculator.dart';
import 'package:ml_algo/src/math/math_analysis/gradient_calculator_impl.dart';

class GradientCalculatorFactory {
  static GradientCalculator defaultGradient() => GradientCalculatorImpl();
}
