import 'package:ml_algo/src/_mixin/data_validation_mixin.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_function.dart';
import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class KnnRegressorImpl with AssessablePredictorMixin, DataValidationMixin
    implements KnnRegressor {

  KnnRegressorImpl(
      this._targetName,
      this._solver,
      this._kernelFn,
      this._dtype,
  );

  final String _targetName;
  final KnnSolver _solver;
  final KernelFn _kernelFn;
  final DType _dtype;

  Vector get _zeroVector => _cachedZeroVector ??= Vector.zero(1, dtype: _dtype);
  Vector _cachedZeroVector;

  @override
  DataFrame predict(DataFrame features) {
    validateTestFeatures(features, _dtype);

    final prediction = Matrix.fromRows(
        _predictOutcomes(features.toMatrix(_dtype))
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
        final weight = _kernelFn(neighbour.distance);
        final weightedLabel = neighbour.label * weight;
        return weightedSum + weightedLabel;
      });

      final weightsSum = kNeighbours.fold<num>(0,
              (sum, neighbour) => sum + _kernelFn(neighbour.distance));

      return weightedLabels / weightsSum;
    });
  }
}
