import 'package:ml_algo/src/common/exception/invalid_metric_type_exception.dart';
import 'package:ml_algo/src/helpers/features_target_split_interface.dart';
import 'package:ml_algo/src/metric/metric_constants.dart';
import 'package:ml_algo/src/metric/metric_factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/model_assessor/model_assessor.dart';
import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

class RegressorAssessor implements ModelAssessor<Predictor> {
  RegressorAssessor(
    this._metricFactory,
    this._featuresTargetSplit,
  );

  final MetricFactory _metricFactory;
  final FeaturesTargetSplit _featuresTargetSplit;

  @override
  double assess(
    Predictor regressor,
    MetricType metricType,
    DataFrame samples,
  ) {
    if (!regressionMetrics.contains(metricType)) {
      throw InvalidMetricTypeException(metricType, regressionMetrics);
    }

    final splits = _featuresTargetSplit(
      samples,
      targetNames: regressor.targetNames,
    ).toList();
    final featuresFrame = splits[0];
    final originalLabelsFrame = splits[1];
    final metric = _metricFactory.createByType(metricType);
    final predictedLabels =
        regressor.predict(featuresFrame).toMatrix(regressor.dtype);
    final originalLabels = originalLabelsFrame.toMatrix(regressor.dtype);

    return metric.getScore(predictedLabels, originalLabels);
  }
}
