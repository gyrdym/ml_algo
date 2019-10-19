import 'package:ml_algo/src/classifier/_mixins/linear_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/helpers/get_probabilities.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class LogisticRegressorImpl with LinearClassifierMixin,
    AssessablePredictorMixin implements LogisticRegressor {

  LogisticRegressorImpl(
      LinearOptimizer _optimizer,
      String className,
      this.linkFunction, {
    bool fitIntercept = false,
    double interceptScale = 1.0,
    Vector initialCoefficients,
    this.dtype = DType.float32,
    this.probabilityThreshold = 0.5,
    num negativeLabel = 0,
    num positiveLabel = 1,
  }) :
        classNames = [className],
        fitIntercept = fitIntercept,
        interceptScale = interceptScale,
        _negativeLabel = negativeLabel,
        _positiveLabel = positiveLabel,
        coefficientsByClasses = _optimizer.findExtrema(
            initialCoefficients: initialCoefficients != null
                ? Matrix.fromColumns([initialCoefficients], dtype: dtype)
                : null,
            isMinimizingObjective: false,
        );

  @override
  final Matrix coefficientsByClasses;

  @override
  final List<String> classNames;

  @override
  final bool fitIntercept;

  @override
  final double interceptScale;

  final DType dtype;

  final double probabilityThreshold;

  final num _positiveLabel;

  final num _negativeLabel;

  @override
  final LinkFunction linkFunction;

  @override
  DataFrame predict(DataFrame features) {
    final processedFeatures = addInterceptIf(
      fitIntercept,
      features.toMatrix(),
      interceptScale,
    );

    final classesList = getProbabilities(
      processedFeatures,
      coefficientsByClasses,
      linkFunction,
    )
    .getColumn(0)
    // TODO: use SIMD
    .map((value) => value >= probabilityThreshold
        ? _positiveLabel
        : _negativeLabel,
    )
    .toList(growable: false);

    final classesMatrix = Matrix.fromColumns([
      Vector.fromList(classesList),
    ]);

    return DataFrame.fromMatrix(
      classesMatrix,
      header: classNames,
    );
  }
}
