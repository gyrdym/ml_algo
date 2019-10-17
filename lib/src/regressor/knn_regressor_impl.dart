import 'package:ml_algo/src/knn_solver/kernel_function/kernel_function.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_function_factory.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_function_factory_impl.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_type.dart';
import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/predictor/assessable_predictor_mixin.dart';
import 'package:ml_algo/src/regressor/knn_regressor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class KnnRegressorImpl with AssessablePredictorMixin implements KnnRegressor {
  KnnRegressorImpl(
      this._trainingFeatures,
      this._trainingOutcomes,
      this._targetName, {
        int k,
        Distance distance = Distance.euclidean,
        FindKnnFn solverFn = findKNeighbours,
        Kernel kernel = Kernel.gaussian,
        DType dtype = DType.float32,

        KernelFunctionFactory kernelFnFactory =
          const KernelFunctionFactoryImpl(),
      }) :
        _k = k,
        _distanceType = distance,
        _solverFn = solverFn,
        _dtype = dtype,
        _kernelFn = kernelFnFactory.createByType(kernel) {
    if (_trainingFeatures.rowsNum != _trainingOutcomes.rowsNum) {
      throw Exception('Number of observations and number of outcomes have to be'
          'equal');
    }
    if (_k > _trainingFeatures.rowsNum) {
      throw Exception('Parameter k should be less than or equal to the number '
          'of training observations');
    }
  }

  final Matrix _trainingFeatures;
  final Matrix _trainingOutcomes;
  final String _targetName;
  final Distance _distanceType;
  final int _k;
  final FindKnnFn _solverFn;
  final KernelFn _kernelFn;
  final DType _dtype;

  Vector get _zeroVector => _cachedZeroVector ??= Vector.zero(
      _trainingOutcomes.columnsNum, dtype: _dtype);
  Vector _cachedZeroVector;

  @override
  DataFrame predict(DataFrame features) {
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
    final neighbours = _solverFn(
      _k,
      _trainingFeatures,
      _trainingOutcomes,
      features,
      standardize: true,
      distance: _distanceType,
    );

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
