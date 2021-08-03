import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/common/json_converter/dtype_json_converter.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type_json_converter.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type_json_converter.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type_json_converter.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type_json_converter_nullable.dart';
import 'package:ml_algo/src/regressor/_mixins/assessable_regressor_mixin.dart';
import 'package:ml_algo/src/regressor/linear_regressor/_injector.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_constants.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_factory.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_json_keys.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

part 'linear_regressor_impl.g.dart';

@JsonSerializable()
@DTypeJsonConverter()
@LinearOptimizerTypeJsonConverter()
@LearningRateTypeJsonConverter()
@InitialCoefficientsTypeJsonConverter()
@RegularizationTypeJsonConverterNullable()
class LinearRegressorImpl
    with AssessableRegressorMixin, SerializableMixin
    implements LinearRegressor {
  LinearRegressorImpl(
    this.coefficients,
    this.targetName,
    this.targetIndex, {
    required this.optimizerType,
    required this.iterationsLimit,
    required this.learningRateType,
    required this.initialCoefficientsType,
    required this.initialLearningRate,
    required this.minCoefficientsUpdate,
    required this.lambda,
    required this.batchSize,
    required this.isFittingDataNormalized,
    bool fitIntercept = false,
    double interceptScale = 1.0,
    this.dtype = DType.float32,
    this.schemaVersion = linearRegressorJsonSchemaVersion,
    this.regularizationType,
    this.randomSeed,
    this.initialCoefficients,
    this.costPerIteration,
  })  : fitIntercept = fitIntercept,
        interceptScale = interceptScale;

  factory LinearRegressorImpl.fromJson(Map<String, dynamic> json) =>
      _$LinearRegressorImplFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$LinearRegressorImplToJson(this);

  @override
  @JsonKey(name: linearRegressorOptimizerTypeJsonKey)
  final LinearOptimizerType optimizerType;

  @override
  @JsonKey(name: linearRegressorIterationsLimitJsonKey)
  final int iterationsLimit;

  @override
  @JsonKey(name: linearRegressorLearningRateTypeJsonKey)
  final LearningRateType learningRateType;

  @override
  @JsonKey(name: linearRegressorInitialCoefficientsTypeJsonKey)
  final InitialCoefficientsType initialCoefficientsType;

  @override
  @JsonKey(name: linearRegressorInitialLearningRateJsonKey)
  final num initialLearningRate;

  @override
  @JsonKey(name: linearRegressorMinCoefficientsUpdateJsonKey)
  final num minCoefficientsUpdate;

  @override
  @JsonKey(name: linearRegressorLambdaJsonKey)
  final num lambda;

  @override
  @JsonKey(name: linearRegressorRegularizationTypeJsonKey)
  final RegularizationType? regularizationType;

  @override
  @JsonKey(name: linearRegressorRandomSeedJsonKey)
  final int? randomSeed;

  @override
  @JsonKey(name: linearRegressorBatchSizeJsonKey)
  final int batchSize;

  @override
  @JsonKey(name: linearRegressorInitialCoefficientsJsonKey)
  final Matrix? initialCoefficients;

  @override
  @JsonKey(name: linearRegressorFittingDataNormalizedFlagJsonKey)
  final bool isFittingDataNormalized;

  @override
  @JsonKey(name: linearRegressorTargetNameJsonKey)
  final String targetName;

  @override
  @JsonKey(name: linearRegressorTargetIndexJsonKey)
  final int targetIndex;

  @override
  @JsonKey(name: linearRegressorFitInterceptJsonKey)
  final bool fitIntercept;

  @override
  @JsonKey(name: linearRegressorInterceptScaleJsonKey)
  final double interceptScale;

  @override
  @JsonKey(name: linearRegressorCoefficientsJsonKey)
  final Vector coefficients;

  @override
  @JsonKey(name: linearRegressorCostPerIterationJsonKey)
  final List<num>? costPerIteration;

  @override
  @JsonKey(name: linearRegressorDTypeJsonKey)
  final DType dtype;

  @override
  @JsonKey(name: jsonSchemaVersionJsonKey)
  final schemaVersion;

  @override
  Iterable<String> get targetNames => [targetName];

  @override
  Iterable<int> get targetIndices => [targetIndex];

  @override
  DataFrame predict(DataFrame features) {
    final splits =
        featuresTargetSplit(features, targetIndices: targetIndices).toList();
    final points = splits[0].toMatrix(dtype);
    final prediction = addInterceptIf(
          fitIntercept,
          points,
          interceptScale,
          dtype,
        ) *
        coefficients;

    return DataFrame.fromMatrix(
      prediction,
      header: targetNames,
    );
  }

  @override
  LinearRegressor retrain(DataFrame data) {
    return linearRegressorInjector.get<LinearRegressorFactory>().create(
          fittingData: data,
          targetName: targetName,
          optimizerType: optimizerType,
          iterationsLimit: iterationsLimit,
          learningRateType: learningRateType,
          initialCoefficientsType: initialCoefficientsType,
          initialLearningRate: initialLearningRate.toDouble(),
          minCoefficientsUpdate: minCoefficientsUpdate.toDouble(),
          lambda: lambda.toDouble(),
          regularizationType: regularizationType,
          fitIntercept: fitIntercept,
          interceptScale: interceptScale,
          randomSeed: randomSeed,
          batchSize: batchSize,
          initialCoefficients: initialCoefficients,
          isFittingDataNormalized: isFittingDataNormalized,
          collectLearningData: false,
          dtype: dtype,
        );
  }
}
