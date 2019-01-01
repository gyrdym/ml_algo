import 'dart:async';

import 'package:ml_algo/float32x4_cross_validator.dart';
import 'package:ml_algo/float32x4_csv_ml_data.dart';
import 'package:ml_algo/gradient_regressor.dart';
import 'package:ml_algo/gradient_type.dart';
import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/metric_type.dart';
import 'package:ml_algo/predictor.dart';
import 'package:tuple/tuple.dart';

Future main() async {
  final data = Float32x4CsvMLData.fromFile('datasets/black_friday.csv',
    labelIdx: 11,
    rows: [const Tuple2(0, 2000)],
    columns: [const Tuple2(2, 11)],
    categories: {
      'Gender': ['M', 'F'],
      'Age': ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'],
      'City_Category': ['A', 'B', 'C'],
      'Stay_In_Current_City_Years': [0, 1, 2, 3, '4+'],
    }
  );
  final features = await data.features;
  final labels = await data.labels;

  final validator = Float32x4CrossValidator.kFold(numberOfFolds: 5);

  final step = 0.000001;
  final start = 0.000555;
  final limit = 0.1;

  double minError = double.infinity;
  double bestLearningRate = 0.0;
  Predictor bestRegressor;

//  for (double rate = start; rate < limit; rate += step) {
  final regressor = GradientRegressor(
      type: GradientType.batch,
      iterationLimit: 100000,
      learningRate: 0.01,
      learningRateType: LearningRateType.decreasing);

  final error = validator.evaluate(regressor, features, labels, MetricType.mape);
  print('Error is: $error');

//    if (error < minError) {
//      minError = error;
//      bestLearningRate = rate;
//      print('Error is: $minError, learning rate: $bestLearningRate');
//    }
//  }
}
