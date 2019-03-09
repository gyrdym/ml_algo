import 'dart:async';

import 'package:ml_algo/ml_algo.dart';

Future main() async {
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

  print('Accuracy is ${accuracy.toStringAsFixed(3)}');
}
