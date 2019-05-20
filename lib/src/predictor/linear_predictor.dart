import 'package:ml_algo/src/predictor/predictor.dart';

abstract class LinearPredictor implements Predictor {
  /// A flag that shows whether the intercept term is fitted or not
  bool get fitIntercept;

  /// A multiplier of intercept term (meaningful only if [fitIntercept] is true)
  double get interceptScale;
}
