import 'package:ml_algo/src/classifier/labels_distribution_calculator/labels_probability_calculator.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';

abstract class LabelsProbabilityCalculatorFactory {
  LabelsProbabilityCalculator create(LinkFunctionType linkFunctionType, Type dtype);
}
