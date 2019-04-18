import 'package:ml_algo/src/algorithms/knn/kernel.dart';
import 'package:ml_algo/src/algorithms/knn/kernel_function_factory.dart';
import 'package:ml_algo/src/algorithms/knn/kernel_function_factory_impl.dart';
import 'package:ml_algo/src/algorithms/knn/kernel_type.dart';
import 'package:ml_algo/src/algorithms/knn/knn.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/regressor/non_parametric_regressor.dart';
import 'package:ml_algo/src/utils/default_parameter_values.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class KNNRegressor implements NoNParametricRegressor {
  KNNRegressor({
    int k,
    Distance distance = Distance.euclidean,
    FindKnnFn solverFn = findKNeighbours,
    Kernel kernel = Kernel.uniform,
    Type dtype = DefaultParameterValues.dtype,

    KernelFunctionFactory kernelFnFactory = const KernelFunctionFactoryImpl(),
  }) :
        _k = k,
        _distanceType = distance,
        _solverFn = solverFn,
        _dtype = dtype,
        _kernelFn = kernelFnFactory.createByType(kernel);

  final Distance _distanceType;
  final int _k;
  final FindKnnFn _solverFn;
  final KernelFn _kernelFn;
  final Type _dtype;

  Matrix _observations;
  Matrix _outcomes;

  Vector get _zeroVector => _cachedZeroVector ??= Vector.zero(
      _outcomes.columnsNum, dtype: _dtype);
  Vector _cachedZeroVector;

  @override
  void fit(Matrix observations, Matrix outcomes, {Matrix initialWeights,
    bool isDataNormalized}) {
    _observations = observations;
    _outcomes = outcomes;
  }

  @override
  Matrix predict(Matrix observations) => Matrix.fromRows(
    _generateOutcomes(observations).toList(growable: false), dtype: _dtype);

  Iterable<Vector> _generateOutcomes(Matrix observations) sync* {
    for (final kNeighbours in _solverFn(_k, _observations, _outcomes,
        observations, distance: _distanceType)) {
      yield kNeighbours
          .fold<Vector>(_zeroVector,
              (sum, pair) => sum + pair.label * _kernelFn(pair.distance)) / _k;
    }
  }

  @override
  double test(Matrix features, Matrix origLabels, MetricType metricType) {
    final metric = MetricFactory.createByType(metricType);
    final prediction = predict(features);
    return metric.getScore(prediction, origLabels);
  }

  @override
  Vector get weights => null;
}
