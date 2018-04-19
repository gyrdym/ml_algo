import 'dart:typed_data';

import 'package:dart_ml/src/metric/metric.dart';
import 'package:dart_ml/src/optimizer/coordinate.dart';
import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:dart_ml/src/regressor/regressor.dart';
import 'package:simd_vector/vector.dart';

class LassoRegressor extends Regressor {
  LassoRegressor({
    List<Float32x4Vector> features,
    Float32List labels,
    int iterationLimit,
    double minWeightsDistance,
    Metric metric,
    double lambda
  }) : super(
    new CoordinateOptimizer(
      InitialWeightsGeneratorFactory.ZeroWeights(),
      minCoefficientsDiff: minWeightsDistance,
      iterationLimit: iterationLimit,
      lambda: lambda
    )
  );
}
