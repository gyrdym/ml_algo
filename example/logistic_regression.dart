import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

void main() {
  final samples = getPimaIndiansDiabetesDataFrame().shuffle();
  final splits = splitData(samples, [0.8]);
  final model = LogisticRegressor(
    splits.first,
    'Outcome',
  );

  print('ACСURACY:');
  print(model.assess(splits.last, MetricType.accuracy));

  print('RECALL:');
  print(model.assess(splits.last, MetricType.recall));

  print('PRECISION:');
  print(model.assess(splits.last, MetricType.precision));

  print('Results (first row - actual values, second row - predicted values):');
  print(splits.last['Outcome'].data
      .take(10)
      .map((val) => num.parse(val.toString()).toDouble()));
  print(model
      .predict(splits.last.dropSeries(names: ['Outcome']))
      .series
      .first
      .data
      .take(10));
}
