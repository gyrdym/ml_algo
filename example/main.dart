import 'dart:async';

import 'gradient_descent_regression.dart';
import 'lasso_regression.dart';
import 'logistic_regression.dart';

Future main() async {
  print('===================================================================');
  await gradientDescentRegression();
  print('===================================================================');
  await lassoRegression();
  print('===================================================================');
  await logisticRegression();
  print('===================================================================');
}