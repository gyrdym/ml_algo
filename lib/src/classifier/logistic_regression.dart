import 'package:dart_ml/src/classifier/classifier.dart';
import 'package:dart_ml/src/metric/classification/type.dart';

class LogisticRegressor extends Classifier {
  LogisticRegressor({
    int iterationLimit,
    double learningRate,
    double minWeightsDistance,
    double alpha,
    double argumentIncrement,
    ClassificationMetricType metric
  }) : super(
    ModuleFactory.logisticRegressionModule(
      learningRate: learningRate,
      minWeightsDistance: minWeightsDistance,
      iterationLimit: iterationLimit,
      metricType: metric,
      lambda: alpha,
      argumentIncrement: argumentIncrement
    )
  );
}
