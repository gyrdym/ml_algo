import 'package:ml_algo/src/optimizer/batch_size_calculator/batch_size_calculator.dart';
import 'package:ml_algo/src/regressor/gradient_type.dart';

class BatchSizeCalculatorImpl implements BatchSizeCalculator {
  const BatchSizeCalculatorImpl();

  @override
  int calculate(GradientType gradientType, int predefinedBatchSize) {
    switch (gradientType) {
      case GradientType.miniBatch:
        return predefinedBatchSize;

      case GradientType.batch:
        return double.maxFinite.floor();

      case GradientType.stochastic:
      default:
        return 1;
    }
  }
}
