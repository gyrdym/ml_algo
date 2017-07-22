part of '../../predictor.dart';

abstract class GradientLinearPredictor extends PredictorBase {
  GradientLinearPredictor({Metric metric}) :
        super(metric: metric ?? new RegressionMetric.RMSE(), scoreFn: new ScoreFunction.Linear());
}
