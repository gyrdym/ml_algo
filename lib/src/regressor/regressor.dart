import 'dart:typed_data';

import 'package:dart_ml/src/metric/factory.dart';
import 'package:dart_ml/src/metric/type.dart';
import 'package:dart_ml/src/model_selection/evaluable.dart';
import 'package:dart_ml/src/optimizer/optimizer.dart';
import 'package:simd_vector/vector.dart';

class Regressor implements Evaluable {

  final Optimizer _optimizer;

  Float32x4Vector get weights => _weights;
  Float32x4Vector _weights;

  Regressor(this._optimizer);

  void fit(
    covariant List<Float32x4Vector> features,
    covariant Float32List labels,
    {
      covariant Float32x4Vector initialWeights
    }
  ) {
    _weights = _optimizer.findExtrema(features, labels, initialWeights: initialWeights, isMinimizingObjective: true);
  }

  double test(
    covariant List<Float32x4Vector> features,
    covariant List<double> origLabels,
    MetricType metricType
  ) {
    final metric = MetricFactory.createByType(metricType);
    final prediction = predict(features);
    return metric.getError(prediction, new Float32x4Vector.from(origLabels));
  }

  Float32x4Vector predict(covariant List<Float32x4Vector> features) {
    final labels = new List<double>(features.length);
    for (int i = 0; i < features.length; i++) {
      labels[i] = _weights.dot(features[i]);
    }
    return new Float32x4Vector.from(labels);
  }
}
