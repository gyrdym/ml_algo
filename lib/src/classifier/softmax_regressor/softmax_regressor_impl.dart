import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/classifier/_mixins/assessable_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/_mixins/linear_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_constants.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_json_keys.dart';
import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/common/json_converter/dtype_json_converter.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/helpers/validate_class_labels.dart';
import 'package:ml_algo/src/helpers/validate_coefficients_matrix.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type_json_converter.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type_json_converter.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type_json_converter.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type_json_converter_nullable.dart';
import 'package:ml_algo/src/link_function/helpers/from_link_function_json.dart';
import 'package:ml_algo/src/link_function/helpers/link_function_to_json.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

part 'softmax_regressor_impl.g.dart';

@JsonSerializable()
@DTypeJsonConverter()
@LinearOptimizerTypeJsonConverter()
@RegularizationTypeJsonConverterNullable()
@LearningRateTypeJsonConverter()
@InitialCoefficientsTypeJsonConverter()
class SoftmaxRegressorImpl
    with
        LinearClassifierMixin,
        AssessableClassifierMixin,
        SerializableMixin
    implements
        SoftmaxRegressor {

  SoftmaxRegressorImpl(
      this.optimizerType,
      this.iterationsLimit,
      this.initialLearningRate,
      this.minCoefficientsUpdate,
      this.lambda,
      this.regularizationType,
      this.randomSeed,
      this.batchSize,
      this.isFittingDataNormalized,
      this.learningRateType,
      this.initialCoefficientsType,
      this.initialCoefficients,
      this.coefficientsByClasses,
      this.targetNames,
      this.linkFunction,
      this.fitIntercept,
      this.interceptScale,
      this.positiveLabel,
      this.negativeLabel,
      this.costPerIteration,
      this.dtype,
      {
        this.schemaVersion = softmaxRegressorJsonSchemaVersion,
      }
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

  @override
  Map<String, dynamic> toJson() => _$SoftmaxRegressorImplToJson(this);

  @override
  @JsonKey(name: softmaxRegressorOptimizerTypeJsonKey)
  final LinearOptimizerType optimizerType;

  @override
  @JsonKey(name: softmaxRegressorIterationsLimitJsonKey)
  final int iterationsLimit;

  @override
  @JsonKey(name: softmaxRegressorInitialLearningRateJsonKey)
  final double initialLearningRate;

  @override
  @JsonKey(name: softmaxRegressorMinCoefsUpdateJsonKey)
  final double minCoefficientsUpdate;

  @override
  @JsonKey(name: softmaxRegressorLambdaJsonKey)
  final double lambda;

  @override
  @JsonKey(name: softmaxRegressorRegularizationTypeJsonKey)
  final RegularizationType? regularizationType;

  @override
  @JsonKey(name: softmaxRegressorRandomSeedJsonKey)
  final int? randomSeed;

  @override
  @JsonKey(name: softmaxRegressorBatchSizeJsonKey)
  final int batchSize;

  @override
  @JsonKey(name: softmaxRegressorFittingDataNormalizedFlagJsonKey)
  final bool isFittingDataNormalized;

  @override
  @JsonKey(name: softmaxRegressorLearningRateTypeJsonKey)
  final LearningRateType learningRateType;

  @override
  @JsonKey(name: softmaxRegressorInitialCoefsTypeJsonKey)
  final InitialCoefficientsType initialCoefficientsType;

  @override
  @JsonKey(name: softmaxRegressorInitialCoefsJsonKey)
  final Matrix? initialCoefficients;

  @override
  @JsonKey(name: softmaxRegressorClassNamesJsonKey)
  final Iterable<String> targetNames;

  @override
  @JsonKey(name: softmaxRegressorFitInterceptJsonKey)
  final bool fitIntercept;

  @override
  @JsonKey(name: softmaxRegressorInterceptScaleJsonKey)
  final num interceptScale;

  @override
  @JsonKey(name: softmaxRegressorCoefficientsByClassesJsonKey)
  final Matrix coefficientsByClasses;

  @override
  @JsonKey(name: softmaxRegressorDTypeJsonKey)
  final DType dtype;

  @override
  @JsonKey(
    name: softmaxRegressorLinkFunctionJsonKey,
    toJson: linkFunctionToJson,
    fromJson: fromLinkFunctionJson,
  )
  final LinkFunction linkFunction;

  @override
  @JsonKey(name: softmaxRegressorPositiveLabelJsonKey)
  final num positiveLabel;

  @override
  @JsonKey(name: softmaxRegressorNegativeLabelJsonKey)
  final num negativeLabel;

  @override
  @JsonKey(name: softmaxRegressorCostPerIterationJsonKey)
  final List<num>? costPerIteration;

  @override
  @JsonKey(name: jsonSchemaVersionJsonKey)
  final schemaVersion;

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
      header: targetNames,
    );
  }

  @override
  SoftmaxRegressor retrain(DataFrame data) {
    return softmaxRegressorInjector
        .get<SoftmaxRegressorFactory>()
        .create(
      trainData: data,
      targetNames: targetNames,
      optimizerType: optimizerType,
      iterationsLimit: iterationsLimit,
      initialLearningRate: initialLearningRate,
      minCoefficientsUpdate: minCoefficientsUpdate,
      lambda: lambda,
      regularizationType: regularizationType,
      randomSeed: randomSeed,
      batchSize: batchSize,
      fitIntercept: fitIntercept,
      interceptScale: interceptScale.toDouble(),
      learningRateType: learningRateType,
      isFittingDataNormalized: isFittingDataNormalized,
      initialCoefficientsType: initialCoefficientsType,
      initialCoefficients: initialCoefficients,
      positiveLabel: positiveLabel,
      negativeLabel: negativeLabel,
      dtype: dtype,
      collectLearningData: false,
    );
  }
}
