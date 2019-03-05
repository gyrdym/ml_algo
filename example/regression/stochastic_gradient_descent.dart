import 'dart:async';

import 'package:ml_algo/ml_algo.dart';
import 'package:tuple/tuple.dart';

Future main() async {
  final data = DataFrame.fromCsv(
    'datasets/black_friday.csv',
    labelIdx: 11,
    rows: [const Tuple2(0, 2999)],
    columns: [const Tuple2(2, 11)],
    categories: {
      'Gender': ['M', 'F'],
      'Age': ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'],
      'City_Category': ['A', 'B', 'C'],
      'Stay_In_Current_City_Years': [0, 1, 2, 3, '4+'],
      'Martial_Status': [0, 1],
      'Product_Category_1': List<int>.generate(19, (i) => i),
      'Product_Category_2': List<int>.generate(19, (i) => i),
      'Product_Category_3': List<int>.generate(19, (i) => i),
    },
    encodeUnknownStrategy: EncodeUnknownValueStrategy.returnZeroes,
  );

  final features = await data.features;
  final labels = await data.labels;

  final validator = CrossValidator.kFold(numberOfFolds: 5);

  final regressor = LinearRegressor.gradient(
      gradientType: GradientType.stochastic,
      iterationsLimit: 300000,
      initialLearningRate: 0.001,
      learningRateType: LearningRateType.constant);

  final error =
      validator.evaluate(regressor, features, labels, MetricType.mape);

  print('MAPE error on k-fold validation: ${error.toStringAsFixed(2)}%');
}
