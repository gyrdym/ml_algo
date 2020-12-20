import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/common/json_converter/dtype_json_converter.dart';
import 'package:ml_algo/src/common/json_converter/matrix_json_converter.dart';
import 'package:ml_algo/src/common/json_converter/vector_json_converter.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type_json_converter.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type_json_converter.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type_json_converter.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/regressor/_mixins/assessable_regressor_mixin.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_json_keys.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

part 'linear_regressor_impl.g.dart';

@JsonSerializable()
@DTypeJsonConverter()
@VectorJsonConverter()
@MatrixJsonConverter()
@LinearOptimizerTypeJsonConverter()
@LearningRateTypeJsonConverter()
@InitialCoefficientsTypeJsonConverter()
class LinearRegressorImpl
    with
        AssessableRegressorMixin,
        SerializableMixin
    implements
        LinearRegressor {

  LinearRegressorImpl(this.coefficients, this.targetName, {
    this.optimizerType,
    this.iterationsLimit,
    this.learningRateType,
    this.initialCoefficientsType,
    this.initialLearningRate,
    this.minCoefficientsUpdate,
    this.lambda,
    this.regularizationType,
    this.randomSeed,
    this.batchSize,
    this.initialCoefficients,
    this.isFittingDataNormalized,
    bool fitIntercept = false,
    double interceptScale = 1.0,
    this.costPerIteration,
    this.dtype = DType.float32,
  }) :
    fitIntercept = fitIntercept,
    interceptScale = interceptScale;

  factory LinearRegressorImpl.fromJson(Map<String, dynamic> json) =>
      _$LinearRegressorImplFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$LinearRegressorImplToJson(this);

  @override
  @JsonKey(name: linearRegressorOptimizerTypeJsonKey)
  final LinearOptimizerType optimizerType;

  @override
  @JsonKey(
    name: linearRegressorIterationsLimitJsonKey,
    includeIfNull: false,
  )
  final int iterationsLimit;

  @override
  @JsonKey(name: linearRegressorLearningRateTypeJsonKey)
  final LearningRateType learningRateType;

  @override
  @JsonKey(
    name: linearRegressorInitialCoefficientsTypeJsonKey,
    includeIfNull: false,
  )
  final InitialCoefficientsType initialCoefficientsType;

  @override
  @JsonKey(name: linearRegressorInitialLearningRateTypeJsonKey)
  final num initialLearningRate;

  @override
  @JsonKey(
    name: linearRegressorMinCoefficientsUpdateJsonKey,
    includeIfNull: false,
  )
  final num minCoefficientsUpdate;

  @override
  @JsonKey(
    name: linearRegressorLambdaJsonKey,
    includeIfNull: false,
  )
  final num lambda;

  @override
  @JsonKey(
    name: linearRegressorRegularizationTypeJsonKey,
    includeIfNull: false,
  )
  final RegularizationType regularizationType;

  @override
  @JsonKey(
    name: linearRegressorRandomSeedJsonKey,
    includeIfNull: false,
  )
  final int randomSeed;

  @override
  @JsonKey(name: linearRegressorBatchSizeJsonKey)
  final int batchSize;

  @override
  @JsonKey(
    name: linearRegressorInitialCoefficientsJsonKey,
    includeIfNull: false,
  )
  final Matrix initialCoefficients;

  @override
  @JsonKey(name: linearRegressorFittingDataNormalizedFlagJsonKey)
  final bool isFittingDataNormalized;

  @override
  @JsonKey(name: linearRegressorTargetNameJsonKey)
  final String targetName;

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
  @JsonKey(
    name: linearRegressorCostPerIterationJsonKey,
    includeIfNull: false,
  )
  final List<num> costPerIteration;

  @override
  @JsonKey(name: linearRegressorDTypeJsonKey)
  final DType dtype;

  @override
  Iterable<String> get targetNames => [targetName];

  @override
  DataFrame predict(DataFrame features) {
    final prediction = addInterceptIf(
      fitIntercept,
      features.toMatrix(dtype),
      interceptScale,
      dtype,
    ) * coefficients;

    return DataFrame.fromMatrix(
        prediction,
        header: targetNames,
    );
  }
}
