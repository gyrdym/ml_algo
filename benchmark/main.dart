import 'dart:async';

import 'gradient_descent_regression.dart';
import 'logistic_regression.dart';
import 'one_hot_encoder.dart';

Future main() async {
  await gradientDescentRegressionBenchmark(); // 0.06 sec (MacBook Air mid 2017) (+0.025)
  await logisticRegressionBenchmark(); // 0.17 sec (MacBook Air mid 2017) (+0.03)
  await oneHotEncoderBenchmark(); // 0.0008 sec (MacBook Air mid 2017)
}
