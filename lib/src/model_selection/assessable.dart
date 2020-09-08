import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

/// An interface for a ML model's performance assessment
abstract class Assessable {
  /// Assesses model performance according to provided [metricType]
  ///
  /// Throws an exception if inappropriate [metricType] provided.
  double assess(DataFrame observations, MetricType metricType);
}
