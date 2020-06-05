import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/helpers/normalize_class_labels.dart';
import 'package:ml_algo/src/helpers/validate_class_labels.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/linalg.dart';

class LogLikelihoodCostFunction implements CostFunction {
  LogLikelihoodCostFunction(
      this._linkFunction,
      this._positiveLabel,
      this._negativeLabel) {
    validateClassLabels(_positiveLabel, _negativeLabel);
  }

  final LinkFunction _linkFunction;
  final num _positiveLabel;
  final num _negativeLabel;

  @override
  double getCost(Matrix x, Matrix w, Matrix y) {
    final normalizedY = normalizeClassLabels(y, _positiveLabel, _negativeLabel);

    return _linkFunction
        .link(x * w)
        .multiply(normalizedY)
        .sum();
  }

  @override
  Matrix getGradient(Matrix x, Matrix w, Matrix y) =>
    x.transpose() * (y.mapRows((labels) =>
        labels.mapToVector(_positiveIndicator)) - _linkFunction.link(x * w));

  @override
  Vector getSubGradient(int wIdx, Matrix x, Matrix w, Matrix y) =>
      throw UnimplementedError();

  double _positiveIndicator(num label) => label == _positiveLabel ? 1 : 0;
}
