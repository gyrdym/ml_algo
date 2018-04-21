import 'dart:typed_data';

import 'package:dart_ml/src/metric/metric.dart';
import 'package:dart_ml/src/optimizer/coordinate_descent.dart';
import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:dart_ml/src/regressor/regressor.dart';
import 'package:simd_vector/vector.dart';

class LassoRegressor extends Regressor {
  LassoRegressor({
    List<Float32x4Vector> features,
    Float32List labels,
    int iterationLimit,
    double minWeightUpdate,
    Metric metric,
    double lambda
  }) : super(
    new CoordinateDescentOptimizer(
      InitialWeightsGeneratorFactory.ZeroWeights(),
      minCoefficientsDiff: minWeightUpdate,
      iterationLimit: iterationLimit,
      lambda: lambda
    )
  );
}
