import 'package:ml_algo/src/default_parameter_values.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory_impl.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/optimizer/coordinate/coordinate.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/regressor/linear_regressor.dart';
import 'package:ml_linalg/linalg.dart';

class LassoRegressor implements LinearRegressor {
  final CoordinateOptimizer _optimizer;
  final InterceptPreprocessor _interceptPreprocessor;

  LassoRegressor({
    // public arguments
    int iterationsLimit = DefaultParameterValues.iterationsLimit,
    double minWeightsUpdate = DefaultParameterValues.minCoefficientsUpdate,
    double lambda,
    bool fitIntercept = false,
    double interceptScale = 1.0,
    Type dtype = DefaultParameterValues.dtype,
    InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,

    // hidden arguments
    InterceptPreprocessorFactory interceptPreprocessorFactory =
        const InterceptPreprocessorFactoryImpl(),
  })  : _interceptPreprocessor = interceptPreprocessorFactory.create(dtype,
            scale: fitIntercept ? interceptScale : 0.0),
        _optimizer = CoordinateOptimizer(
          initialWeightsType: initialWeightsType,
          costFunctionType: CostFunctionType.squared,
          iterationsLimit: iterationsLimit,
          minCoefficientsDiff: minWeightsUpdate,
          lambda: lambda,
          dtype: dtype,
        );

  @override
  MLVector get weights => _weights;
  MLVector _weights;

  @override
  void fit(MLMatrix features, MLVector labels,
      {MLVector initialWeights, bool isDataNormalized = false}) {
    _weights = _optimizer
        .findExtrema(
          _interceptPreprocessor.addIntercept(features),
          MLMatrix.columns([labels]),
          initialWeights: initialWeights != null
              ? MLMatrix.rows([initialWeights])
              : null,
          isMinimizingObjective: true,
          arePointsNormalized: isDataNormalized
        ).getRow(0);
  }

  @override
  double test(MLMatrix features, MLVector origLabels, MetricType metricType) {
    final metric = MetricFactory.createByType(metricType);
    final prediction = predict(features);
    return metric.getError(prediction, origLabels);
  }

  MLVector predict(MLMatrix features) => (features * _weights).toVector();
}
