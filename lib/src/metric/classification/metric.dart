import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_linalg/vector.dart';

abstract class ClassificationMetric implements Metric {
  @override
  double getError(MLVector predictedLabels, MLVector origLabels);

  double getScore(MLVector predictedLabels, MLVector origLabels);
}
