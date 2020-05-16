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
import 'package:ml_linalg/dtype_to_json.dart';
import 'package:ml_linalg/from_dtype_json.dart';
import 'package:ml_linalg/from_matrix_json.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/matrix_to_json.dart';
import 'package:ml_linalg/vector.dart';

part 'logistic_regressor_impl.g.dart';

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
  @JsonKey(
    name: logisticRegressorCoefficientsByClassesJsonKey,
    toJson: matrixToJson,
    fromJson: fromMatrixJson,
  )
  final Matrix coefficientsByClasses;

  @override
  @JsonKey(name: logisticRegressorClassNamesJsonKey)
  final Iterable<String> classNames;

  @override
  @JsonKey(name: logisticRegressorFitInterceptJsonKey)
  final bool fitIntercept;

  @override
  @JsonKey(name: logisticRegressorInterceptScaleJsonKey)
  final num interceptScale;

  @override
  @JsonKey(
    name: logisticRegressorDTypeJsonKey,
    toJson: dTypeToJson,
    fromJson: fromDTypeJson,
  )
  final DType dtype;

  @JsonKey(name: logisticRegressorProbabilityThresholdJsonKey)
  final num probabilityThreshold;

  @JsonKey(name: logisticRegressorPositiveLabelJsonKey)
  final num positiveLabel;

  @JsonKey(name: logisticRegressorNegativeLabelJsonKey)
  final num negativeLabel;

  @override
  @JsonKey(
    name: logisticRegressorLinkFunctionJsonKey,
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
