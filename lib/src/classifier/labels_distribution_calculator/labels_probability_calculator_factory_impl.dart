import 'package:ml_algo/src/classifier/labels_distribution_calculator/labels_probability_calculator.dart';
import 'package:ml_algo/src/classifier/labels_distribution_calculator/labels_probability_calculator_factory.dart';
import 'package:ml_algo/src/classifier/labels_distribution_calculator/labels_probability_calculator_impl.dart';

class LabelsProbabilityCalculatorFactoryImpl implements LabelsProbabilityCalculatorFactory {
  const LabelsProbabilityCalculatorFactoryImpl();

  @override
  LabelsProbabilityCalculator create(linkFunction, Type dtype) => LabelsProbabilityCalculatorImpl(linkFunction, dtype);
}
