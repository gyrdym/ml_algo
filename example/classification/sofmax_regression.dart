import 'dart:async';

import 'package:ml_algo/ml_algo.dart';
import 'package:tuple/tuple.dart';

Future main() async {
  final data = DataFrame.fromCsv('datasets/iris.csv',
    labelIdx: 5,
    columns: [const Tuple2<int, int>(1, 5)],
    categoryNameToEncoder: {
      'Species': CategoricalDataEncoderType.oneHot,
    },
  );

  final features = await data.features;
  final labels = await data.labels;

  final validator = CrossValidator.kFold(numberOfFolds: 5);

  final softmaxRegressor = LinearClassifier.softmaxRegressor(
      initialLearningRate: 0.03,
      iterationsLimit: null,
      minWeightsUpdate: 1e-6,
      randomSeed: 46,
      learningRateType: LearningRateType.constant);

  final accuracy = validator.evaluate(
      softmaxRegressor, features, labels, MetricType.accuracy);

  print('Accuracy is ${accuracy.toStringAsFixed(2)}');
}
