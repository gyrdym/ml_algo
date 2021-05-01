import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/model_selection/model_assessor/model_assessor.dart';
import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

mixin AssessableRegressorMixin implements Assessable, Predictor {
  @override
  double assess(
    DataFrame samples,
    MetricType metricType,
  ) =>
      injector
          .get<ModelAssessor<Predictor>>()
          .assess(this, metricType, samples);
}
