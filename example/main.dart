import 'dart:async';

import 'gradient_descent_regression.dart';
import 'lasso_regression.dart';
import 'logistic_regression.dart';

Future main() async {
  print('Learning in process, wait a bit...');
  print('\n');

  final sgdQuality = await gradientDescentRegression();
  print('========================================================================================================');
  print('|| Stochastic gradient descent regression, K-fold cross validation with MAPE metric (error in percents):');
  print('|| ${sgdQuality.toStringAsFixed(2)}%');
  print('========================================================================================================');

  print('\n');

  final lassoQuality = await lassoRegression();
  print('========================================================================================================');
  print('|| Lasso regression, K-fold cross validation with MAPE metric (error in percent):');
  print('|| ${lassoQuality.toStringAsFixed(2)}%');
  print('=========================================================================================================');

  print('\n');
  print('Learning of logistic regressor in progress, wait a bit... (best possible parameters are being fitted)');
  print('\n');

  final logisticRegressionError = await logisticRegression();
  print('=========================================================================================================');
  print('|| Logistic regression, error on cross validation: ');
  print('|| ${(logisticRegressionError * 100).toStringAsFixed(2)}%,');
  print('=========================================================================================================');
}
