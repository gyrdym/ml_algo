import 'dart:async';

import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/linear_regressor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

/// A simple usage example using synthetic data. To see more complex examples, please, visit other directories in this
/// folder
Future main() async {
  // Let's create a feature matrix (a set of independent variables)
  final features = MLMatrix.from([
    [2.0, 3.0, 4.0, 5.0],
    [12.0, 32.0, 1.0, 3.0],
    [27.0, 3.0, 0.0, 59.0],
  ]);

  // Let's create dependent variables vector. It will be used as `true` values to adjust regression coefficients
  final labels = MLVector.from([4.3, 3.5, 2.1]);

  // Let's create a regressor itself. With its help we can train some linear model to predict a label value for a new
  // features
  final regressor = LinearRegressor.gradient(
      iterationsLimit: 100,
      initialLearningRate: 0.0005,
      learningRateType: LearningRateType.constant);

  // Let's train our model (training or fitting is a coefficients adjusting process)
  regressor.fit(features, labels);

  // Let's see adjusted coefficients
  print('Regression coefficients: ${regressor.weights}');
}
