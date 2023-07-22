import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

void main() {
  final samples = getHousingDataFrame().shuffle(seed: 4);
  final splits = splitData(samples, [0.8]);
  final model = LinearRegressor(splits.first, 'MEDV',
      optimizerType: LinearOptimizerType.gradient,
      batchSize: splits.first.rows.length,
      collectLearningData: true,
      // learningRateType: LearningRateType.timeBased,
      initialLearningRate: 0.01,
      decay: 290000);

  print('MAPE error: ${model.assess(splits.last, MetricType.mape)}');
  print('Cost per iteration: ${model.costPerIteration}');
  print(model
      .predict(splits.last.dropSeries(names: ['MEDV']))['MEDV']
      .data
      .take(30)
      .map((val) => num.parse(val.toString()).toStringAsFixed(1))
      .toList());
  print(splits.last['MEDV'].data.take(30).toList());
}
