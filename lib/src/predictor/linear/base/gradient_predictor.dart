import 'package:dart_ml/src/predictor/base/predictor_impl.dart';
import 'package:dart_ml/src/optimizer/gradient/base.dart';
import 'package:dart_ml/src/metric/metric.dart';
import 'package:dart_ml/src/score_function/score_function.dart';

abstract class GradientLinearPredictor extends PredictorImpl {
  GradientLinearPredictor(GradientOptimizer optimizer, {Metric metric}) :
        super(optimizer, metric: metric ?? new RegressionMetric.RMSE(), scoreFn: new ScoreFunction.Linear());
}
