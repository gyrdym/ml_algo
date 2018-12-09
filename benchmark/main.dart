import 'dart:async';

import 'gradient_descent_regression.dart';
import 'logistic_regression.dart';

Future main() async {
  await gradientDescentRegression(); // 0.065 sec (MacBook Air mid 2017) (+0.03)
  await logisticRegression(); // 0.19 sec (MacBook Air mid 2017) (+0.05)
}
