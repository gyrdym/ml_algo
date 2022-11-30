import 'package:ml_linalg/matrix.dart';

/// A simple usage example using synthetic data. To see more complex examples,
/// please, visit other directories in this folder
Future<void> main() async {
  final features = Matrix.fromList([
    [2, 2],
    [3, 3],
    [4, 4],
    [5, 5],
  ]);
  final labels = Matrix.column([12, 18, 24, 30]);
  final initialCoefficients = Matrix.column([0, 0]);

  final coefficients = gradientDescent(features, labels, initialCoefficients);

  print('Coefficients: $coefficients');
}

Matrix gradientDescent(Matrix X, Matrix Y, Matrix initialCoefficients) {
  final learningRate = 1e-3;
  final iterationLimit = 50;

  var coefficients = initialCoefficients;

  var coefficientDiff = 1e10;
  var minCoefficientDiff = 1e-5;

  for (var i = 0; i < iterationLimit; i++) {
    if (coefficientDiff <= minCoefficientDiff) {
      break;
    }

    final gradient = X.transpose() * -2 * (Y - X * coefficients);
    final newCoefficients = coefficients - gradient * learningRate;

    coefficientDiff = (newCoefficients - coefficients).norm();
    coefficients = newCoefficients;
  }

  return coefficients;
}
