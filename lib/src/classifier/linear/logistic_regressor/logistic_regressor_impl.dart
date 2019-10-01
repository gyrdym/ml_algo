import 'package:ml_algo/src/classifier/linear/linear_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/linear/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/helpers/get_probabilities.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class LogisticRegressorImpl with LinearClassifierMixin,
    AssessablePredictorMixin implements LogisticRegressor {

  LogisticRegressorImpl(
      LinearOptimizer _optimizer,
      this._className,
      this.classLabels,
      this.linkFunction, {
    bool fitIntercept = false,
    double interceptScale = 1.0,
    Matrix initialWeights,
    this.dtype = DType.float32,
    this.probabilityThreshold = 0.5,
  }) :
        fitIntercept = fitIntercept,
        interceptScale = interceptScale,
        coefficientsByClasses = _optimizer.findExtrema(
            initialCoefficients: initialWeights,
            isMinimizingObjective: false,
        );

  @override
  final Matrix coefficientsByClasses;

  final String _className;

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
  DataFrame predict(Matrix features) {
    final processedFeatures = addInterceptIf(
      fitIntercept,
      features,
      interceptScale,
    );

    final classesList = getProbabilities(
      processedFeatures,
      coefficientsByClasses,
      linkFunction,
    )
    .getColumn(0)
    // TODO: use SIMD
    .map((value) => value >= probabilityThreshold ? 1 : 0)
    .toList(growable: false);

    final classesMatrix = Matrix.fromColumns([
      Vector.fromList(classesList),
    ]);

    return DataFrame.fromMatrix(
      classesMatrix,
      header: [_className],
    );
  }
}
