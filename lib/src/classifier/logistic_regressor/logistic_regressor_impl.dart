import 'package:ml_algo/src/classifier/_mixins/linear_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/helpers/validate_coefficients_matrix.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class LogisticRegressorImpl with LinearClassifierMixin,
    AssessablePredictorMixin implements LogisticRegressor {

  LogisticRegressorImpl(
      String className,
      this.linkFunction,
      this.fitIntercept,
      this.interceptScale,
      this.coefficientsByClasses,
      this._probabilityThreshold,
      this._negativeLabel,
      this._positiveLabel,
      this.dtype,
  ) : classNames = [className] {
    validateCoefficientsMatrix(coefficientsByClasses);

    // Logistic regression specific check, it cannot be placed in
    // `validateCoefficientsMatrix`
    if (coefficientsByClasses.columnsNum > 1) {
      throw Exception('Expected to have coefficients just for a single class, '
          'but coefficients for ${coefficientsByClasses.columnsNum} classes '
          'provided. Please, check your linear optimizer implementation '
          '(Logistic Regression deals only with single class problem)');
    }
  }

  /// N x 1 matrix, where N - number of features. It has only one column since
  /// in case of Logistic Regression only one class is used
  @override
  final Matrix coefficientsByClasses;

  @override
  final List<String> classNames;

  @override
  final bool fitIntercept;

  @override
  final num interceptScale;

  @override
  final LinkFunction linkFunction;

  @override
  final DType dtype;

  final num _probabilityThreshold;
  final num _positiveLabel;
  final num _negativeLabel;

  @override
  DataFrame predict(DataFrame testFeatures) {
    final probabilities = getProbabilitiesMatrix(testFeatures).getColumn(0);

    final classesList = probabilities
        // TODO: use SIMD
        .map((value) => value >= _probabilityThreshold
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
