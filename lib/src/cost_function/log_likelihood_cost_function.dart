import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/helpers/normalize_class_labels.dart';
import 'package:ml_algo/src/helpers/validate_class_labels.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/linalg.dart';

class LogLikelihoodCostFunction implements CostFunction {
  LogLikelihoodCostFunction(
      this._linkFunction, this._positiveLabel, this._negativeLabel) {
    validateClassLabels(_positiveLabel, _negativeLabel);
  }

  final LinkFunction _linkFunction;
  final num _positiveLabel;
  final num _negativeLabel;

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
  Vector getSubGradient(int wIdx, Matrix x, Matrix w, Matrix y) =>
      throw UnimplementedError('Coordinate optimization is not implemented yet '
          'for log likelihood cost function');
}
