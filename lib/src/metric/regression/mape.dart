import 'package:ml_algo/src/metric/regression/metric.dart';
import 'package:ml_linalg/linalg.dart';

class MAPEMetric implements RegressionMetric {
  const MAPEMetric();

  @override
  double getError(MLVector predictedLabels, MLVector origLabels) =>
      100 /
      predictedLabels.length *
      ((origLabels - predictedLabels) / origLabels).abs().sum();
}
