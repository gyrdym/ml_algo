import 'dart:typed_data';

import 'package:dart_ml/src/metric/factory.dart';
import 'package:dart_ml/src/metric/type.dart';
import 'package:dart_ml/src/model_selection/evaluable.dart';
import 'package:dart_ml/src/optimizer/optimizer.dart';
import 'package:simd_vector/vector.dart';

class Classifier implements Evaluable {

  final Optimizer _optimizer;

  Classifier(this._optimizer);

  @override
  void fit(
    covariant List<Float32x4Vector> features,
    covariant List<double> origLabels,
    {
      covariant Float32x4Vector initialWeights
    }
  ) {
    _optimizer.findExtrema(features, origLabels, initialWeights: initialWeights);
  }

  @override
  double test(
    covariant List<Float32x4Vector> features,
    covariant List<double> origLabels,
    MetricType metricType
  ) {
    final metric = MetricFactory.createByType(metricType);
    final prediction = predictClasses(features);
    return metric.getError(prediction, new Float32x4Vector.from(origLabels));
  }

  Float32x4Vector predictClasses(List<Float32x4Vector> features) {
    Float32List probabilities = predict(features).asList();
    return new Float32x4Vector.from(probabilities.map((double value) => value.round() * 1.0));
  }
}
