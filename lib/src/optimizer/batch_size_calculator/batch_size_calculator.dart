import 'package:ml_algo/gradient_type.dart';

abstract class BatchSizeCalculator {
  int calculate(GradientType gradientType, int predefinedBatchSize);
}
