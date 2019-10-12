import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

abstract class Assessable {
  /// Assesses model according to provided [metricType]
  double assess(DataFrame observations, Iterable<String> targetNames,
      MetricType metricType);
}
