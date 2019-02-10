import 'package:ml_algo/src/classifier/labels_probability_calculator/labels_probability_calculator.dart';
import 'package:ml_algo/src/classifier/labels_probability_calculator/labels_probability_calculator_factory.dart';
import 'package:ml_algo/src/classifier/labels_probability_calculator/labels_probability_calculator_impl.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';

class LabelsProbabilityCalculatorFactoryImpl implements LabelsProbabilityCalculatorFactory {
  const LabelsProbabilityCalculatorFactoryImpl();

  @override
  LabelsProbabilityCalculator create(LinkFunctionType linkFunctionType, Type dtype) =>
      LabelsProbabilityCalculatorImpl(linkFunctionType, dtype);
}
