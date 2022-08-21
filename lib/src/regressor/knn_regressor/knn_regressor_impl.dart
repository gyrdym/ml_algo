import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/common/json_converter/dtype_json_converter.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/helpers/validate_test_features.dart';
import 'package:ml_algo/src/knn_kernel/kernel.dart';
import 'package:ml_algo/src/knn_kernel/kernel_json_converter.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_json_converter.dart';
import 'package:ml_algo/src/regressor/_mixins/assessable_regressor_mixin.dart';
import 'package:ml_algo/src/regressor/knn_regressor/_injector.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_constants.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_factory.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_json_keys.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

part 'knn_regressor_impl.g.dart';

final _float32zeroVector = Vector.zero(1, dtype: DType.float32);
final _float64zeroVector = Vector.zero(1, dtype: DType.float64);

@JsonSerializable()
@KnnSolverJsonConverter()
@KernelJsonConverter()
@DTypeJsonConverter()
class KnnRegressorImpl
    with AssessableRegressorMixin, SerializableMixin
    implements KnnRegressor {
  KnnRegressorImpl(
    this.targetName,
    this.solver,
    this.kernel,
    this.dtype, {
    this.schemaVersion = knnRegressorJsonSchemaVersion,
  });

  factory KnnRegressorImpl.fromJson(Map<String, dynamic> json) =>
      _$KnnRegressorImplFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$KnnRegressorImplToJson(this);

  @override
  int get k => solver.k;

  @override
  KernelType get kernelType => kernel.type;

  @override
  Distance get distanceType => solver.distanceType;

  @override
  @JsonKey(name: knnRegressorDTypeJsonKey)
  final DType dtype;

  @JsonKey(name: knnRegressorTargetNameJsonKey)
  final String targetName;

  @JsonKey(name: knnRegressorSolverJsonKey)
  final KnnSolver solver;

  @JsonKey(name: knnRegressorKernelJsonKey)
  final Kernel kernel;

  @override
  Iterable<String> get targetNames => [targetName];

  @override
  @JsonKey(name: jsonSchemaVersionJsonKey)
  final int? schemaVersion;

  Vector get _zeroVector =>
      dtype == DType.float32 ? _float32zeroVector : _float64zeroVector;

  @override
  DataFrame predict(DataFrame testFeatures) {
    validateTestFeatures(testFeatures, dtype);

    final prediction = Matrix.fromRows(
      _predictOutcomes(testFeatures.toMatrix(dtype)).toList(growable: false),
      dtype: dtype,
    );

    return DataFrame.fromMatrix(
      prediction,
      header: [targetName],
    );
  }

  @override
  KnnRegressor retrain(DataFrame data) {
    return knnRegressorInjector.get<KnnRegressorFactory>().create(
          data,
          targetName,
          k,
          kernelType,
          distanceType,
          dtype,
        );
  }

  Iterable<Vector> _predictOutcomes(Matrix features) {
    final neighbours = solver.findKNeighbours(features);

    return neighbours.map((kNeighbours) {
      final weightedLabels =
          kNeighbours.fold<Vector>(_zeroVector, (weightedSum, neighbour) {
        final weight = kernel.getWeightByDistance(neighbour.distance);
        final weightedLabel = neighbour.label * weight;

        return weightedSum + weightedLabel;
      });

      final weightsSum = kNeighbours.fold<num>(
          0,
          (sum, neighbour) =>
              sum + kernel.getWeightByDistance(neighbour.distance));

      return weightedLabels / weightsSum;
    });
  }
}
