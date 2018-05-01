import 'package:dart_ml/src/metric/factory.dart';
import 'package:dart_ml/src/metric/type.dart';
import 'package:dart_ml/src/model_selection/evaluable.dart';
import 'package:dart_ml/src/optimizer/optimizer.dart';
import 'package:simd_vector/vector.dart';

abstract class Classifier implements Evaluable {

  final Optimizer _optimizer;
  final int _numberOfClasses;

  final List<Float32x4Vector> _weightsByClass;

  Classifier(this._numberOfClasses, this._optimizer) :
    _weightsByClass = new List<Float32x4Vector>(_numberOfClasses);

  @override
  void fit(
    covariant List<Float32x4Vector> features,
    covariant List<double> origLabels,
    {
      covariant Float32x4Vector initialWeights,
      bool isDataNormalized
    }
  ) {
    for (int classId = 0; classId < _numberOfClasses; classId++) {
      final labels = _makeLabelsOneVsAll(origLabels, classId);
      _weightsByClass[classId] = _optimizer.findExtrema(features, labels,
        initialWeights: initialWeights,
        arePointsNormalized: isDataNormalized
      );
    }
  }

  Float32x4Vector _makeLabelsOneVsAll(Float32x4Vector origLabels, int targetLabel) =>
    new Float32x4Vector.from(origLabels.map((int label) => label == targetLabel ? 1 : -1));

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

  List<Float32x4Vector> predictProbabilities(List<Float32x4Vector> points) {
    List<Float32x4Vector> probabilitiesByClass = new List<Float32x4Vector>(_numberOfClasses);

    for (int i = 0; i < points.length; i++) {
      final labels = new List<double>(points.length);
        for (int classId = 0; classId < _numberOfClasses; classId++) {
          labels[classId] = _weightsByClass[classId].dot(points[i]);
        }
        probabilitiesByClass[i] = labels;
    }

    return probabilitiesByClass;
  }

  Float32x4Vector predictClasses(List<Float32x4Vector> features) {
    final probabilities = predictProbabilities(features);
    final classes = new Float32x4Vector.zero(features.length);
    for (int i = 0; i< probabilities.length; i++) {
      final _probabilities = probabilities[i];
      classes[i] = _probabilities.indexOf(_probabilities.max()) * 1.0;
    }
    return classes;
  }
}
