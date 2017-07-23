part of 'package:dart_ml/src/predictor/predictor.dart';

abstract class _GradientLinearPredictor extends _PredictorBase {
  _GradientLinearPredictor({Metric metric}) :
        super(metric: metric ?? new RegressionMetric.RMSE(), scoreFn: new ScoreFunction.Linear());
}
