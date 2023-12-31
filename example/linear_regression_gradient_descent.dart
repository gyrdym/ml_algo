import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

void main() async {
  final sourceSamples = await fromCsv('example/fish.csv');
  final encoder = Encoder.oneHot(sourceSamples, columnNames: ['Species']);
  final samples = encoder.process(sourceSamples).shuffle(seed: 13);
  final splits = splitData(samples, [0.8]);
  final targetColumn = 'Weight';
  final model = LinearRegressor(splits.first, targetColumn,
      optimizerType: LinearOptimizerType.gradient,
      batchSize: splits.first.rows.length,
      initialLearningRate: 1e-9,
  );

  print('Error: ${model.assess(splits.last, MetricType.mape)}');
}
