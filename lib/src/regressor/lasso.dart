import 'package:dart_ml/src/cost_function/cost_function_factory.dart';
import 'package:dart_ml/src/optimizer/coordinate.dart';
import 'package:dart_ml/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:dart_ml/src/regressor/linear_regressor.dart';

class LassoRegressor extends LinearRegressor {
  LassoRegressor({
    int iterationLimit,
    double minWeightUpdate,
    double lambda,
    bool fitIntercept = false,
    double interceptScale = 1.0
  }) : super(
    new CoordinateOptimizer(
      InitialWeightsGeneratorFactory.zeroWeights(),
      CostFunctionFactory.squared(),
      minCoefficientsDiff: minWeightUpdate,
      iterationLimit: iterationLimit,
      lambda: lambda
    ),
    fitIntercept ? interceptScale : 0.0
  );
}
