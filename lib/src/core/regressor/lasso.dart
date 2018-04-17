import 'package:dart_ml/src/core/metric/metric.dart';
import 'package:dart_ml/src/core/regressor/regressor_impl.dart';
import 'package:dart_ml/src/core/score_function/score_function.dart';
import 'package:dart_ml/src/di/lasso_regressor_module.dart';

class LassoRegressor extends RegressorImpl {
  LassoRegressor({
    int iterationLimit,
    double minWeightsDistance,
    Metric metric,
    ScoreFunction scoreFn,
    double lambda
  }) : super(createLassoRegressionModule(
      iterationLimit: iterationLimit,
      minWeightsDistance: minWeightsDistance,
      metric: metric,
      scoreFn: scoreFn,
      lambda: lambda
  ));
}
