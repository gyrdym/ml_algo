import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/helpers/normalize_class_labels.dart';
import 'package:ml_algo/src/helpers/validate_class_labels.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class LogLikelihoodCostFunction implements CostFunction {
  LogLikelihoodCostFunction(this._linkFunction, this._positiveLabel,
      this._negativeLabel, this._dtype) {
    validateClassLabels(_positiveLabel, _negativeLabel);
  }

  final LinkFunction _linkFunction;
  final num _positiveLabel;
  final num _negativeLabel;
  final DType _dtype;

  @override
  double getCost(Matrix x, Matrix w, Matrix y) {
    final positiveY =
        y.mapElements((label) => label == _positiveLabel ? 1.0 : 1e-10);
    final negativeY =
        y.mapElements((label) => label == _negativeLabel ? 1.0 : 1e-10);
    final positiveProbabilities = _linkFunction.link(x * w);
    final negativeProbabilities =
        positiveProbabilities.mapElements((probability) => 1 - probability);
    final onlyPositive = positiveProbabilities.multiply(positiveY);
    final onlyNegative = negativeProbabilities.multiply(negativeY);

    return (onlyNegative + onlyPositive).log().sum();
  }

  @override
  Matrix getGradient(Matrix x, Matrix w, Matrix y) {
    final yNormalized = normalizeClassLabels(y, _positiveLabel, _negativeLabel);

    return x.transpose() * (yNormalized - _linkFunction.link(x * w));
  }

  @override
  Matrix getHessian(Matrix x, Matrix w, Matrix y) {
    final prediction = _linkFunction.link(x * w).toVector();
    final ones = Vector.filled(x.rowsNum, 1.0, dtype: _dtype);
    final V = Matrix.diagonal((prediction * (ones - prediction)).toList(),
        dtype: _dtype);

    return x.transpose() * V * x;
  }
}
