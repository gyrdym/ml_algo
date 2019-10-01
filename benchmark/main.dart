import 'linear_regressor.dart' as gradient_descent_regression_benchmark;
import 'logistic_regressor.dart' as logistic_regression_benchmark;
import 'algorithms/knn.dart' as knn_regressor_benchmark;

Future main() async {
  //  (MacBook Air mid 2017)
  await gradient_descent_regression_benchmark.main(); // 0.07 sec
  await logistic_regression_benchmark.main(); // 0.12 sec
  await knn_regressor_benchmark.main(); // 5 sec
}
