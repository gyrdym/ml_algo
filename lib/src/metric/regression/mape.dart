import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_linalg/linalg.dart';

class MAPEMetric implements Metric {
  const MAPEMetric();

  @override
  double getScore(MLVector predictedLabels, MLVector origLabels) =>
      100 /
      predictedLabels.length *
      ((origLabels - predictedLabels) / origLabels).abs().sum();
}
