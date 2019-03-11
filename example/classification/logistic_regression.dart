import 'dart:async';

import 'package:ml_algo/ml_algo.dart';
import 'package:tuple/tuple.dart';

Future binaryClassification() async {
  final data = DataFrame.fromCsv('datasets/pima_indians_diabetes_database.csv',
    labelName: 'class variable (0 or 1)',
  );

  final features = (await data.features)
      .mapColumns((vector) => vector / vector.norm());
  final labels = await data.labels;
  final validator = CrossValidator.kFold(numberOfFolds: 5);
  final logisticRegressor = LinearClassifier.logisticRegressor(
      initialLearningRate: .8,
      iterationsLimit: 500,
      gradientType: GradientType.batch,
      fitIntercept: true,
      interceptScale: .1,
      learningRateType: LearningRateType.constant);

  final accuracy = validator.evaluate(
      logisticRegressor,
      features,
      labels,
      MetricType.accuracy);

  print('Pima indians diabetes dataset, binary classification: accuracy is '
      '${accuracy.toStringAsFixed(3)}');
}

Future multiclassClassification() async {
  final data = DataFrame.fromCsv('datasets/iris.csv',
    labelName: 'Species',
    columns: [const Tuple2(1, 5)],
    categoryNameToEncoder: {
      'Species': CategoricalDataEncoderType.oneHot,
    },
  );

  final features = await data.features;
  final labels = await data.labels;

  final validator = CrossValidator.kFold(numberOfFolds: 5);

  final softmaxRegressor = LinearClassifier.logisticRegressor(
      initialLearningRate: 0.03,
      iterationsLimit: 200,
      minWeightsUpdate: 1e-6,
      randomSeed: 46,
      learningRateType: LearningRateType.constant);

  final accuracy = validator.evaluate(
      softmaxRegressor, features, labels, MetricType.accuracy);

  print('Iris dataset, multiclass avr classification: accuracy is '
      '${accuracy.toStringAsFixed(2)}');
}

Future main() async {
  print('Logistic regression');
  await binaryClassification();
  await multiclassClassification();
}
