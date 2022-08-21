import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

void main() {
  final samples = getPimaIndiansDiabetesDataFrame().shuffle();
  final splits = splitData(samples, [0.8]);
  final model = LogisticRegressor(
    splits.first,
    'Outcome',
    batchSize: splits.first.rows.length,
    learningRateType: LearningRateType.exponential,
    decay: 0.7,
    collectLearningData: true,
  );

  print('ACURACY:');
  print(model.assess(splits.last, MetricType.accuracy));

  print('RECALL:');
  print(model.assess(splits.last, MetricType.recall));

  print('PRECISION:');
  print(model.assess(splits.last, MetricType.precision));

  print('LD: ');
  print(splits.last['Outcome'].data.take(10));
  print(model
      .predict(splits.last.dropSeries(names: ['Outcome']))
      .series
      .first
      .data
      .take(10));
}
