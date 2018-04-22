import 'package:dart_ml/src/metric/factory.dart';
import 'package:dart_ml/src/metric/type.dart';
import 'package:dart_ml/src/model_selection/evaluable.dart';
import 'package:dart_ml/src/optimizer/optimizer.dart';
import 'package:simd_vector/vector.dart';

class Classifier implements Evaluable {

  final Optimizer _optimizer;

  Float32x4Vector _weights;

  Classifier(this._optimizer);

  @override
  void fit(
    covariant List<Float32x4Vector> features,
    covariant List<double> origLabels,
    {
      covariant Float32x4Vector initialWeights,
      bool isDataNormalized
    }
  ) {
    _weights = _optimizer.findExtrema(features, origLabels,
      initialWeights: initialWeights,
      arePointsNormalized: isDataNormalized
    );
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

  Float32x4Vector predictProbabilities(List<Float32x4Vector> features) {
    final labels = new List<double>(features.length);
    for (int i = 0; i < features.length; i++) {
      labels[i] = _weights.dot(features[i]);
    }
    return new Float32x4Vector.from(labels);
  }

  Float32x4Vector predictClasses(List<Float32x4Vector> features) {
    final probabilities = predictProbabilities(features).asList();
    return new Float32x4Vector.from(probabilities.map((double value) => value.round() * 1.0));
  }
}
