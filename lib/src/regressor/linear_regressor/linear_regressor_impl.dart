import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_json_keys.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/dtype_to_json.dart';
import 'package:ml_linalg/from_dtype_json.dart';
import 'package:ml_linalg/from_vector_json.dart';
import 'package:ml_linalg/vector.dart';
import 'package:ml_linalg/vector_to_json.dart';

part 'linear_regressor_impl.g.dart';

@JsonSerializable()
class LinearRegressorImpl
    with
        AssessablePredictorMixin,
        SerializableMixin
    implements
        LinearRegressor {

  LinearRegressorImpl(this.coefficients, this.targetName, {
    bool fitIntercept = false,
    double interceptScale = 1.0,
    this.dtype = DType.float32,
  }) :
    fitIntercept = fitIntercept,
    interceptScale = interceptScale;

  factory LinearRegressorImpl.fromJson(Map<String, dynamic> json) =>
      _$LinearRegressorImplFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$LinearRegressorImplToJson(this);

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
  @JsonKey(
    name: linearRegressorCoefficientsJsonKey,
    toJson: vectorToJson,
    fromJson: fromVectorJson,
  )
  final Vector coefficients;

  @override
  @JsonKey(
    name: linearRegressorDTypeJsonKey,
    toJson: dTypeToJson,
    fromJson: fromDTypeJson,
  )
  final DType dtype;

  @override
  DataFrame predict(DataFrame features) {
    final prediction = addInterceptIf(
      fitIntercept,
      features.toMatrix(),
      interceptScale,
    ) * coefficients;

    return DataFrame.fromMatrix(
        prediction,
        header: [targetName],
    );
  }
}
