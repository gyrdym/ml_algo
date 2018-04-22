import 'package:dart_ml/src/optimizer/coordinate_descent.dart';
import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:dart_ml/src/regressor/regressor.dart';

class LassoRegressor extends Regressor {
  LassoRegressor({
    int iterationLimit,
    double minWeightUpdate,
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
