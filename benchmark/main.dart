import 'dart:async';

import 'gradient_descent_regression.dart';
import 'logistic_regression.dart';

Future main() async {
  await gradientDescentRegression(); // 0.06 sec (MacBook Air mid 2017) (+0.025)
  await logisticRegression(); // 0.17 sec (MacBook Air mid 2017) (+0.03)
}
