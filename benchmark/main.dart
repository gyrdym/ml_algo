import 'dart:async';

import 'gradient_descent_regression.dart' as gradientDescentRegressionBenchmark;
import 'logistic_regression.dart' as logisticRegressionBenchmark;
import 'algorithms/knn.dart' as knnBenchmark;

Future main() async {
  //  (MacBook Air mid 2017)
  await gradientDescentRegressionBenchmark.main(); // 0.07 sec
  await logisticRegressionBenchmark.main(); // 0.12 sec
  await knnBenchmark.main(); // 5 sec
}
