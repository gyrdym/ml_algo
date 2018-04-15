import 'dart:typed_data';
import 'package:dart_ml/src/core/metric/factory.dart';
import 'package:dart_ml/src/core/metric/metric.dart';
import 'package:dart_ml/src/core/metric/type.dart';
import 'package:dart_ml/src/core/predictor/predictor_base.dart';
import 'package:simd_vector/vector.dart';

class ClassifierBase extends PredictorBase {
  @override
  double test(List<Float32x4Vector> features, List<double> origLabels, {MetricType metric}) {
    Metric _metric = metric == null ? metric : MetricFactory.createByType(metric);
    Float32x4Vector prediction = predictClasses(features);
    return _metric.getError(prediction, new Float32x4Vector.from(origLabels));
  }

  Float32x4Vector predictClasses(List<Float32x4Vector> features) {
    Float32List probabilities = predict(features).asList();
    return new Float32x4Vector.from(probabilities.map((double value) => value.round() * 1.0));
  }
}
