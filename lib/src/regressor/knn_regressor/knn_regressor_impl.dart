import 'package:ml_algo/src/helpers/validate_test_features.dart';
import 'package:ml_algo/src/knn_kernel/kernel.dart';
import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class KnnRegressorImpl with AssessablePredictorMixin implements KnnRegressor {
  KnnRegressorImpl(
      this._targetName,
      this._solver,
      this._kernel,
      this._dtype,
  );

  final String _targetName;
  final KnnSolver _solver;
  final Kernel _kernel;
  final DType _dtype;

  Vector get _zeroVector => _cachedZeroVector ??= Vector.zero(1, dtype: _dtype);
  Vector _cachedZeroVector;

  @override
  DataFrame predict(DataFrame testFeatures) {
    validateTestFeatures(testFeatures, _dtype);

    final prediction = Matrix.fromRows(
        _predictOutcomes(testFeatures.toMatrix(_dtype))
            .toList(growable: false),
        dtype: _dtype,
    );

    return DataFrame.fromMatrix(
      prediction,
      header: [_targetName],
    );
  }

  Iterable<Vector> _predictOutcomes(Matrix features) {
    final neighbours = _solver.findKNeighbours(features);

    return neighbours.map((kNeighbours) {
      final weightedLabels = kNeighbours.fold<Vector>(_zeroVector, (weightedSum, neighbour) {
        final weight = _kernel.getWeightByDistance(neighbour.distance);
        final weightedLabel = neighbour.label * weight;
        return weightedSum + weightedLabel;
      });

      final weightsSum = kNeighbours.fold<num>(0,
              (sum, neighbour) => sum + _kernel.getWeightByDistance(neighbour.distance));

      return weightedLabels / weightsSum;
    });
  }
}
