import 'dart:async';

import 'gradient_descent_regression.dart';
import 'logistic_regression.dart';
import 'one_hot_encoder.dart';

Future main() async {
  //  (MacBook Air mid 2017)
  await gradientDescentRegressionBenchmark(); // 0.0034 sec
  await logisticRegressionBenchmark(); // 0.0015 sec
  await oneHotEncoderBenchmark(); // 1.1 sec
}
