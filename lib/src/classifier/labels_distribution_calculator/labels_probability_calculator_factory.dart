import 'package:ml_algo/src/classifier/labels_distribution_calculator/labels_probability_calculator.dart';
import 'package:ml_algo/src/link_function/link_function.dart';

abstract class LabelsProbabilityCalculatorFactory {
  LabelsProbabilityCalculator create(LinkFunction linkFunction, Type dtype);
}
