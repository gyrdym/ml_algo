import 'package:ml_algo/src/helpers/validate_test_features.dart';
import 'package:ml_algo/src/knn_kernel/kernel.dart';
import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/regressor/_mixins/assessable_regressor_mixin.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class KnnRegressorImpl
    with
        AssessableRegressorMixin
    implements
        KnnRegressor {
  KnnRegressorImpl(
      this._targetName,
      this._solver,
      this._kernel,
      this.dtype,
  );

  @override
  final DType dtype;

  @override
  Iterable<String> get targetNames => [_targetName];

  final String _targetName;
  final KnnSolver _solver;
  final Kernel _kernel;

  Vector get _zeroVector => _cachedZeroVector ??= Vector.zero(1, dtype: dtype);
  Vector _cachedZeroVector;

  @override
  DataFrame predict(DataFrame testFeatures) {
    validateTestFeatures(testFeatures, dtype);

    final prediction = Matrix.fromRows(
        _predictOutcomes(testFeatures.toMatrix(dtype))
            .toList(growable: false),
        dtype: dtype,
    );

    return DataFrame.fromMatrix(
      prediction,
      header: [_targetName],
    );
  }

  Iterable<Vector> _predictOutcomes(Matrix features) {
    final neighbours = _solver.findKNeighbours(features);

    return neighbours.map((kNeighbours) {
      final weightedLabels = kNeighbours.fold<Vector>(_zeroVector,
              (weightedSum, neighbour) {
        final weight = _kernel.getWeightByDistance(neighbour.distance);
        final weightedLabel = neighbour.label * weight;

        return weightedSum + weightedLabel;
      });

      final weightsSum = kNeighbours.fold<num>(0,
              (sum, neighbour) => sum + _kernel
                  .getWeightByDistance(neighbour.distance));

      return weightedLabels / weightsSum;
    });
  }
}
