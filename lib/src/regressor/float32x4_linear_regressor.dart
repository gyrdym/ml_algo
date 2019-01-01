import 'dart:typed_data';

import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/metric_type.dart';
import 'package:ml_algo/predictor.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_linalg/linalg.dart';

abstract class Float32x4LinearRegressor implements Predictor<Float32x4> {
  final Optimizer<Float32x4> _optimizer;
  final InterceptPreprocessor _interceptPreprocessor;

  Float32x4LinearRegressor(this._optimizer, double interceptScale)
      : _interceptPreprocessor = InterceptPreprocessor(interceptScale: interceptScale);

  MLVector<Float32x4> get weights => _weights;
  MLVector<Float32x4> _weights;

  @override
  void fit(MLMatrix<Float32x4> features, MLVector<Float32x4> labels,
      {MLVector<Float32x4> initialWeights, bool isDataNormalized = false}) {
    _weights = _optimizer.findExtrema(_interceptPreprocessor.addIntercept(features), labels,
        initialWeights: initialWeights, isMinimizingObjective: true, arePointsNormalized: isDataNormalized);
  }

  @override
  double test(MLMatrix<Float32x4> features, MLVector<Float32x4> origLabels, MetricType metricType) {
    final metric = MetricFactory.createByType(metricType);
    final prediction = predict(features);
    return metric.getError(prediction, origLabels);
  }

  MLVector<Float32x4> predict(MLMatrix<Float32x4> features) {
    final labels = List<double>(features.rowsNum);
    for (int i = 0; i < features.rowsNum; i++) {
      labels[i] = _weights.dot(features.getRow(i));
    }
    return Float32x4Vector.from(labels);
  }
}
