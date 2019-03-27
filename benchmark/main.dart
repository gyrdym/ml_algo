import 'dart:async';

import 'gradient_descent_regression.dart';
import 'logistic_regression.dart';

Future main() async {
  //  (MacBook Air mid 2017)
  await gradientDescentRegressionBenchmark(); // 0.05 sec
  await logisticRegressionBenchmark(); // 0.11 sec
}
