import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_linalg/linalg.dart';

abstract class RegressionMetric implements Metric {
  @override
  double getError(MLVector predictedLabels, MLVector origLabels);
}
