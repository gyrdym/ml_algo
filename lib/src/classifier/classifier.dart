import 'package:dart_ml/src/metric/factory.dart';
import 'package:dart_ml/src/metric/type.dart';
import 'package:dart_ml/src/model_selection/evaluable.dart';
import 'package:dart_ml/src/optimizer/optimizer.dart';
import 'package:simd_vector/vector.dart';

abstract class Classifier implements Evaluable {

  final Optimizer _optimizer;
  final int _numberOfClasses;

  final List<Float64x2Vector> _weightsByClasses;

  Classifier(this._numberOfClasses, this._optimizer) :
    _weightsByClasses = new List<Float64x2Vector>(_numberOfClasses);

  @override
  void fit(
    covariant List<Float64x2Vector> features,
    covariant List<double> origLabels,
    {
      covariant Float64x2Vector initialWeights,
      bool isDataNormalized
    }
  ) {
    for (int classId = 0; classId < _numberOfClasses; classId++) {
      final labels = _makeLabelsOneVsAll(origLabels, classId);
      _weightsByClasses[classId] = _optimizer.findExtrema(features, labels,
        initialWeights: initialWeights,
        arePointsNormalized: isDataNormalized
      );
    }
  }

  Float64x2Vector _makeLabelsOneVsAll(Float64x2Vector origLabels, int targetLabel) =>
    new Float64x2Vector.from(origLabels.map((double label) => label == targetLabel ? 1.0 : 0.0));

  @override
  double test(
    covariant List<Float64x2Vector> features,
    covariant List<double> origLabels,
    MetricType metricType
  ) {
    final metric = MetricFactory.createByType(metricType);
    final prediction = predictClasses(features);
    return metric.getError(prediction, new Float64x2Vector.from(origLabels));
  }

  List<Float64x2Vector> predictProbabilities(List<Float64x2Vector> points) {
    final distribution = new List<Float64x2Vector>(points.length);

    for (int i = 0; i < points.length; i++) {
      final probabilities = new List<double>(_numberOfClasses);
      for (int classId = 0; classId < _numberOfClasses; classId++) {
        final score = _weightsByClasses[classId].dot(points[i]);
        probabilities[classId] = _optimizer.costFunction.linkScoreToProbability(score);
      }
      distribution[i] = new Float64x2Vector.from(probabilities);
      print(distribution[i].sum());
    }
    return distribution;
  }

  Float64x2Vector predictClasses(List<Float64x2Vector> features) {
    final probabilities = predictProbabilities(features);
    final classes = new List<double>(features.length);
    for (int i = 0; i < probabilities.length; i++) {
      final _probabilities = probabilities[i];
      classes[i] = _probabilities.indexOf(_probabilities.max()) * 1.0;
    }
    return new Float64x2Vector.from(classes);
  }
}
