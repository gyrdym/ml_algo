import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/model_selection/model_assessor/model_assessor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

mixin AssessableClassifierMixin implements Assessable, Classifier {
  @override
  double assess(
    DataFrame samples,
    MetricType metricType,
  ) =>
      module
          .get<ModelAssessor<Classifier>>()
          .assess(this, metricType, samples);
}
