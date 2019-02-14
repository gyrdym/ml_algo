import 'package:ml_algo/src/regressor/gradient_type.dart';

abstract class BatchSizeCalculator {
  int calculate(GradientType gradientType, int predefinedBatchSize);
}
