import 'package:dart_ml/src/core/classifier/classifier_impl.dart';
import 'package:dart_ml/src/core/metric/classification/type.dart';
import 'package:dart_ml/src/core/optimizer/regularization.dart';
import 'package:dart_ml/src/di/factory.dart';

class LogisticRegressor extends ClassifierImpl {
  LogisticRegressor({int iterationLimit, double learningRate, double minWeightsDistance, double alpha,
                      double argumentIncrement, ClassificationMetricType metric, Regularization regularization}) :

        super(ModuleFactory.logisticRegressionModule(learningRate: learningRate,
                                                               minWeightsDistance: minWeightsDistance,
                                                               iterationLimit: iterationLimit,
                                                               metricType: metric,
                                                               regularization: regularization,
                                                               lambda: alpha,
                                                               argumentIncrement: argumentIncrement));
}
