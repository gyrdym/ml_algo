import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

Future<void> main() async {
  final features = DataFrame([
    ['feature_1', 'feature_2', 'output'],
    [2, 2, 12],
    [3, 3, 18],
    [4, 4, 24],
    [5, 5, 30],
  ]);
  final model = LinearRegressor.SGD(features, 'output', fitIntercept: false);

  print('Coefficients: ${model.coefficients}');
}
