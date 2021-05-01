import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator.dart';
import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator_factory.dart';
import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator_impl.dart';

class DistributionCalculatorFactoryImpl
    implements DistributionCalculatorFactory {
  const DistributionCalculatorFactoryImpl();

  @override
  DistributionCalculator create() => const DistributionCalculatorImpl();
}
