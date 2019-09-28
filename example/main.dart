import 'dart:async';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';

/// A simple usage example using synthetic data. To see more complex examples,
/// please, visit other directories in this folder
Future main() async {
  // Let's create a dataframe with fitting data, let's assume, that the target
  // column is the fifth column (column with index 4)
  final dataFrame = DataFrame(<Iterable<double>>[
    [2.0, 3.0, 4.0, 5.0, 4.3],
    [12.0, 32.0, 1.0, 3.0, 3.5],
    [27.0, 3.0, 0.0, 59.0, 2.1],
  ], headerExists: false);

  // Let's create a regressor itself and train it
  final regressor = LinearRegressor.gradient(
      dataFrame, 'col_4',
      iterationsLimit: 100,
      initialLearningRate: 0.0005,
      learningRateType: LearningRateType.constant);

  // Let's see adjusted coefficients
  print('Regression coefficients: ${regressor.coefficients}');
}
