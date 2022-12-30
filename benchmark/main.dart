import 'linear_regressor.dart' as gradient_descent_regression_benchmark;
import 'logistic_regressor_gradient.dart' as logistic_regression_benchmark;
import 'knn_solver.dart' as knn_regressor_benchmark;

Future<void> main() async {
  //  (MacBook Air mid 2017)
  await gradient_descent_regression_benchmark.main(); // 0.07 sec
  await logistic_regression_benchmark.main(); // 0.12 sec
  knn_regressor_benchmark.main(); // 5 sec
}
