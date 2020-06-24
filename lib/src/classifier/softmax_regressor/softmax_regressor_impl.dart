import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/classifier/_mixins/linear_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_json_keys.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/helpers/validate_class_labels.dart';
import 'package:ml_algo/src/helpers/validate_coefficients_matrix.dart';
import 'package:ml_algo/src/link_function/helpers/from_link_function_json.dart';
import 'package:ml_algo/src/link_function/helpers/link_function_to_json.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:ml_linalg/matrix.dart';

part 'softmax_regressor_impl.g.dart';

@JsonSerializable()
class SoftmaxRegressorImpl
    with
        LinearClassifierMixin,
        AssessablePredictorMixin,
        SerializableMixin
    implements
        SoftmaxRegressor {

  SoftmaxRegressorImpl(
      this.coefficientsByClasses,
      this.classNames,
      this.linkFunction,
      this.fitIntercept,
      this.interceptScale,
      this.positiveLabel,
      this.negativeLabel,
      this.costPerIteration,
      this.dtype,
  ) {
    validateClassLabels(positiveLabel, negativeLabel);
    validateCoefficientsMatrix(coefficientsByClasses);

    // Softmax regression specific check, it cannot be placed in
    // `validateCoefficientsMatrix`
    if (coefficientsByClasses.columnsNum < 2) {
      throw Exception('Expected coefficients at least for two classes. '
          'Please, check your linear optimizer implementation or the way your '
          'data was encoded');
    }
  }

  factory SoftmaxRegressorImpl.fromJson(Map<String, dynamic> json) =>
      _$SoftmaxRegressorImplFromJson(json);

  Map<String, dynamic> toJson() => _$SoftmaxRegressorImplToJson(this);

  @override
  @JsonKey(name: softmaxRegressorClassNamesJsonKey)
  final Iterable<String> classNames;

  @override
  @JsonKey(name: softmaxRegressorFitInterceptJsonKey)
  final bool fitIntercept;

  @override
  @JsonKey(name: softmaxRegressorInterceptScaleJsonKey)
  final num interceptScale;

  @override
  @JsonKey(
    name: softmaxRegressorCoefficientsByClassesJsonKey,
    toJson: matrixToJson,
    fromJson: fromMatrixJson,
  )
  final Matrix coefficientsByClasses;

  @override
  @JsonKey(
    name: softmaxRegressorDTypeJsonKey,
    toJson: dTypeToJson,
    fromJson: fromDTypeJson,
  )
  final DType dtype;

  @override
  @JsonKey(
    name: softmaxRegressorLinkFunctionJsonKey,
    toJson: linkFunctionToJson,
    fromJson: fromLinkFunctionJson,
  )
  final LinkFunction linkFunction;

  @JsonKey(name: softmaxRegressorPositiveLabelJsonKey)
  final num positiveLabel;

  @JsonKey(name: softmaxRegressorNegativeLabelJsonKey)
  final num negativeLabel;

  @override
  @JsonKey(
    name: softmaxRegressorCostPerIterationJsonKey,
    includeIfNull: false,
  )
  final List<num> costPerIteration;

  @override
  DataFrame predict(DataFrame testFeatures) {
    final allProbabilities = getProbabilitiesMatrix(testFeatures);
    final labels = allProbabilities.mapRows((probabilities) {
      final positiveLabelIdx = probabilities
          .toList()
          .indexOf(probabilities.max());
      final predictedRow = List.filled(
        coefficientsByClasses.columnsNum,
        negativeLabel,
      );

      predictedRow[positiveLabelIdx] = positiveLabel;

      return Vector.fromList(predictedRow, dtype: dtype);
    });

    return DataFrame.fromMatrix(
      labels,
      header: classNames,
    );
  }
}
