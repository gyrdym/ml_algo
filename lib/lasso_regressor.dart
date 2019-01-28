import 'package:ml_algo/src/cost_function/cost_function_factory_impl.dart';
import 'package:ml_algo/src/optimizer/coordinate.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory_impl.dart';
import 'package:ml_algo/src/regressor/float32x4_linear_regressor.dart';

class LassoRegressor extends Float32x4LinearRegressor {
  LassoRegressor(
      {int iterationLimit,
      double minWeightUpdate,
      double lambda,
      bool fitIntercept = false,
      double interceptScale = 1.0})
      : super(
            CoordinateOptimizer(InitialWeightsGeneratorFactoryImpl.zeroes(), CostFunctionFactoryImpl.squared(),
                minCoefficientsDiff: minWeightUpdate, iterationLimit: iterationLimit, lambda: lambda),
            fitIntercept ? interceptScale : 0.0);
}
