import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/classifier/_mixins/assessable_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/_mixins/linear_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_constants.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_factory.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_json_keys.dart';
import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/common/exception/outdated_json_schema_exception.dart';
import 'package:ml_algo/src/common/json_converter/dtype_json_converter.dart';
import 'package:ml_algo/src/common/json_converter/matrix_json_converter.dart';
import 'package:ml_algo/src/common/json_converter/matrix_json_converter_nullable.dart';
import 'package:ml_algo/src/common/json_converter/vector_json_converter.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/helpers/validate_class_labels.dart';
import 'package:ml_algo/src/helpers/validate_coefficients_matrix.dart';
import 'package:ml_algo/src/helpers/validate_probability_threshold.dart';
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

part 'logistic_regressor_impl.g.dart';

@JsonSerializable()
@LinearOptimizerTypeJsonConverter()
@DTypeJsonConverter()
@MatrixJsonConverter()
@MatrixJsonConverterNullable()
@VectorJsonConverter()
@RegularizationTypeJsonConverterNullable()
@LearningRateTypeJsonConverter()
@InitialCoefficientsTypeJsonConverter()
class LogisticRegressorImpl
    with
        LinearClassifierMixin,
        AssessableClassifierMixin,
        SerializableMixin
    implements
        LogisticRegressor {

  LogisticRegressorImpl(
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
      {
        this.schemaVersion = logisticRegressorJsonSchemaVersion,
      }
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

  @override
  @JsonKey(name: logisticRegressorLinearOptimizerTypeJsonKey)
  final LinearOptimizerType optimizerType;

  @override
  @JsonKey(
    name: logisticRegressorIterationsLimitJsonKey,
    includeIfNull: false,
  )
  final int iterationsLimit;

  @override
  @JsonKey(
    name: logisticRegressorInitialLearningRateJsonKey,
    includeIfNull: false,
  )
  final double initialLearningRate;

  @override
  @JsonKey(
    name: logisticRegressorMinCoefsUpdateJsonKey,
    includeIfNull: false,
  )
  final double minCoefficientsUpdate;

  @override
  @JsonKey(
    name: logisticRegressorLambdaJsonKey,
    includeIfNull: false,
  )
  final double lambda;

  @override
  @JsonKey(
    name: logisticRegressorRegularizationTypeJsonKey,
    includeIfNull: false,
  )
  final RegularizationType? regularizationType;

  @override
  @JsonKey(
    name: logisticRegressorRandomSeedJsonKey,
    includeIfNull: false,
  )
  final int? randomSeed;

  @override
  @JsonKey(name: logisticRegressorBatchSizeJsonKey)
  final int batchSize;

  @override
  @JsonKey(name: logisticRegressorDataNormalizedFlagJsonKey)
  final bool isFittingDataNormalized;

  @override
  @JsonKey(name: logisticRegressorLearningRateTypeJsonKey)
  final LearningRateType learningRateType;

  @override
  @JsonKey(
    name: logisticRegressorInitCoefficientsTypeJsonKey,
    includeIfNull: false,
  )
  final InitialCoefficientsType initialCoefficientsType;

  @override
  @JsonKey(
    name: logisticRegressorInitCoefficientsJsonKey,
    includeIfNull: false,
  )
  final Vector? initialCoefficients;

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

  @override
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
  final List<num>? costPerIteration;

  @override
  @JsonKey(name: jsonSchemaVersionJsonKey)
  final schemaVersion;

  final _outdatedSchemaVersions = [null];

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

  @override
  LogisticRegressor retrain(DataFrame data) {
    if (_outdatedSchemaVersions.contains(schemaVersion)) {
      throw OutdatedJsonSchemaException();
    }

    return logisticRegressorInjector
        .get<LogisticRegressorFactory>()
        .create(
      trainData: data,
      targetName: targetNames.first,
      optimizerType: optimizerType,
      iterationsLimit: iterationsLimit,
      initialLearningRate: initialLearningRate,
      minCoefficientsUpdate: minCoefficientsUpdate,
      probabilityThreshold: probabilityThreshold.toDouble(),
      lambda: lambda,
      regularizationType: regularizationType,
      randomSeed: randomSeed,
      batchSize: batchSize,
      fitIntercept: fitIntercept,
      interceptScale: interceptScale.toDouble(),
      isFittingDataNormalized: isFittingDataNormalized,
      learningRateType: learningRateType,
      initialCoefficientsType: initialCoefficientsType,
      initialCoefficients: initialCoefficients ?? Vector.empty(dtype: dtype),
      positiveLabel: positiveLabel,
      negativeLabel: negativeLabel,
      collectLearningData: false,
      dtype: dtype,
    );
  }
}
