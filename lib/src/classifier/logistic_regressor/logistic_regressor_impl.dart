import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/classifier/_mixins/assessable_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/_mixins/linear_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_json_keys.dart';
import 'package:ml_algo/src/common/json_converter/dtype_json_converter.dart';
import 'package:ml_algo/src/common/json_converter/matrix_json_converter.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/helpers/validate_class_labels.dart';
import 'package:ml_algo/src/helpers/validate_coefficients_matrix.dart';
import 'package:ml_algo/src/helpers/validate_probability_threshold.dart';
import 'package:ml_algo/src/link_function/helpers/from_link_function_json.dart';
import 'package:ml_algo/src/link_function/helpers/link_function_to_json.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

part 'logistic_regressor_impl.g.dart';

@JsonSerializable()
@DTypeJsonConverter()
@MatrixJsonConverter()
class LogisticRegressorImpl
    with
        LinearClassifierMixin,
        AssessableClassifierMixin,
        SerializableMixin
    implements
        LogisticRegressor {

  LogisticRegressorImpl(
      this.targetNames,
      this.linkFunction,
      this.fitIntercept,
      this.interceptScale,
      this.coefficientsByClasses,
      this.probabilityThreshold,
      this.negativeLabel,
      this.positiveLabel,
      this.costPerIteration,
      this.dtype,
  ) {
    validateProbabilityThreshold(probabilityThreshold);
    validateClassLabels(positiveLabel, negativeLabel);
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

  factory LogisticRegressorImpl.fromJson(Map<String, dynamic> json) =>
      _$LogisticRegressorImplFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$LogisticRegressorImplToJson(this);

  /// N x 1 matrix, where N - number of features. It has only one column since
  /// in case of Logistic Regression only one class is used
  @override
  @JsonKey(name: logisticRegressorCoefficientsByClassesJsonKey)
  final Matrix coefficientsByClasses;

  @override
  @JsonKey(name: logisticRegressorClassNamesJsonKey)
  final Iterable<String> targetNames;

  @override
  @JsonKey(name: logisticRegressorFitInterceptJsonKey)
  final bool fitIntercept;

  @override
  @JsonKey(name: logisticRegressorInterceptScaleJsonKey)
  final num interceptScale;

  @override
  @JsonKey(name: logisticRegressorDTypeJsonKey)
  final DType dtype;

  @JsonKey(name: logisticRegressorProbabilityThresholdJsonKey)
  final num probabilityThreshold;

  @override
  @JsonKey(name: logisticRegressorPositiveLabelJsonKey)
  final num positiveLabel;

  @override
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
  @JsonKey(
    name: logisticRegressorCostPerIterationJsonKey,
    includeIfNull: false,
  )
  final List<num> costPerIteration;

  @override
  DataFrame predict(DataFrame testFeatures) {
    final predictedLabels = getProbabilitiesMatrix(testFeatures)
        .mapColumns(
            (column) => column.mapToVector(
                    (probability) => probability >= probabilityThreshold
                        ? positiveLabel.toDouble()
                        : negativeLabel.toDouble()
            ),
    );

    return DataFrame.fromMatrix(
      predictedLabels,
      header: targetNames,
    );
  }
}
