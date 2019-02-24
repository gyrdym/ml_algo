import 'dart:async';

import 'package:ml_algo/ml_algo.dart';
import 'package:tuple/tuple.dart';

Future main() async {
  final data = MLData.fromCsvFile('datasets/iris.csv',
    labelIdx: 5,
    columns: [const Tuple2<int, int>(1, 5)],
    encoderType: CategoricalDataEncoderType.ordinal,
    categories: {
      'Species': ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
    },
  );

  final features = await data.features;
  final labels = await data.labels;

  final validator = CrossValidator.kFold(numberOfFolds: 5);

  final softmaxRegressor = LinearClassifier.softmaxRegressor(
      initialLearningRate: 0.00053,
      iterationsLimit: 500,
      minWeightsUpdate: null,
      randomSeed: 46,
      learningRateType: LearningRateType.constant);

  final accuracy = validator.evaluate(
      softmaxRegressor, features, labels, MetricType.accuracy);

  print('Error is ${(accuracy * 100).toStringAsFixed(2)}%');
}
