part of 'package:dart_ml/src/predictor/predictor.dart';

abstract class GradientLinearPredictor extends PredictorBase {
  GradientLinearPredictor({Metric metric}) :
        super(metric: metric ?? new RegressionMetric.RMSE(), scoreFn: new ScoreFunction.Linear());
}
