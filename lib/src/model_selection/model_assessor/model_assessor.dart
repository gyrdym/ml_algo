import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

abstract class ModelAssessor<T extends Predictor> {
  double assess(
      T predictor,
      MetricType metricType,
      DataFrame samples,
  );
}
