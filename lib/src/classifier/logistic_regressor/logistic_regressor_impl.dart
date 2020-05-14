import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/classifier/_mixins/linear_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_json_keys.dart';
import 'package:ml_algo/src/helpers/validate_coefficients_matrix.dart';
import 'package:ml_algo/src/link_function/helpers/from_link_function_json.dart';
import 'package:ml_algo/src/link_function/helpers/link_function_to_json.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

part 'logistic_regressor.g.dart';

@JsonSerializable()
class LogisticRegressorImpl with LinearClassifierMixin,
    AssessablePredictorMixin implements LogisticRegressor {

  LogisticRegressorImpl(
      this.classNames,
      this.linkFunction,
      this.fitIntercept,
      this.interceptScale,
      this.coefficientsByClasses,
      this.probabilityThreshold,
      this.negativeLabel,
      this.positiveLabel,
      this.dtype,
  ) {
    validateCoefficientsMatrix(coefficientsByClasses);

    // Logistic regression specific check, it cannot be placed in
    // `validateCoefficientsMatrix`
    if (coefficientsByClasses.columnsNum > 1) {
      throw Exception('Expected coefficients for a single class, but '
          'coefficients for ${coefficientsByClasses.columnsNum} classes '
          'provided. Please, check your linear optimizer implementation '
          '(Logistic Regression deals only with single class problem)');
    }
  }

  /// N x 1 matrix, where N - number of features. It has only one column since
  /// in case of Logistic Regression only one class is used
  @override
  @JsonKey(name: coefficientsByClassesJsonKey)
  final Matrix coefficientsByClasses;

  @override
  @JsonKey(name: classNamesJsonKey)
  final Iterable<String> classNames;

  @override
  @JsonKey(name: fitInterceptJsonKey)
  final bool fitIntercept;

  @override
  @JsonKey(name: interceptScaleJsonKey)
  final num interceptScale;

  @override
  @JsonKey(name: dTypeJsonKey)
  final DType dtype;

  @JsonKey(name: probabilityThresholdJsonKey)
  final num probabilityThreshold;

  @JsonKey(name: positiveLabelJsonKey)
  final num positiveLabel;

  @JsonKey(name: negativeLabelJsonKey)
  final num negativeLabel;

  @override
  @JsonKey(
    name: linkFunctionJsonKey,
    toJson: linkFunctionToJson,
    fromJson: fromLinkFunctionJson,
  )
  final LinkFunction linkFunction;

  @override
  DataFrame predict(DataFrame testFeatures) {
    final probabilities = getProbabilitiesMatrix(testFeatures).getColumn(0);

    final classesList = probabilities
        // TODO: use SIMD
        .map((value) => value >= probabilityThreshold
          ? positiveLabel
          : negativeLabel,
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
