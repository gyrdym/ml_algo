import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator.dart';
import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator_factory.dart';
import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator_impl.dart';

class SequenceElementsDistributionCalculatorFactoryImpl implements
    SequenceElementsDistributionCalculatorFactory {

  const SequenceElementsDistributionCalculatorFactoryImpl();

  @override
  SequenceElementsDistributionCalculator create() =>
      const SequenceElementsDistributionCalculatorImpl();
}
