import 'package:ml_algo/src/classifier/linear/linear_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/linear/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/helpers/get_probabilities.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/logit/inverse_logit_link_function.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_algo/src/solver/linear/linear_optimizer.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class LogisticRegressorImpl with LinearClassifierMixin,
    AssessablePredictorMixin implements LogisticRegressor {

  LogisticRegressorImpl(LinearOptimizer _optimizer, this.classLabels, {
    bool fitIntercept = false,
    double interceptScale = 1.0,
    Matrix initialWeights,
    this.dtype = DType.float32,
    this.probabilityThreshold = 0.5,
  }) :
        fitIntercept = fitIntercept,
        interceptScale = interceptScale,
        linkFunction = InverseLogitLinkFunction(dtype),
        coefficientsByClasses = _optimizer.findExtrema(
            initialWeights: initialWeights,
            isMinimizingObjective: false,
        );

  @override
  final Matrix coefficientsByClasses;

  @override
  final Matrix classLabels;

  @override
  final bool fitIntercept;

  @override
  final double interceptScale;

  final DType dtype;

  final double probabilityThreshold;

  @override
  final LinkFunction linkFunction;

  @override
  Matrix predict(Matrix features) {
    final processedFeatures = addInterceptIf(fitIntercept, features,
        interceptScale);
    final classesSource = getProbabilities(processedFeatures,
        coefficientsByClasses, linkFunction)
        .getColumn(0)
        // TODO: use SIMD
        .map((value) => value >= probabilityThreshold ? 1.0 : 0.0)
        .toList(growable: false);
    return Matrix.fromColumns([Vector.fromList(classesSource)]);
  }
}
