import 'package:dart_ml/src/core/metric/metric.dart';
import 'package:dart_ml/src/core/metric/type.dart';
import 'package:dart_ml/src/core/predictor/predictor.dart';
import 'package:dart_ml/src/core/predictor/predictor_base.dart';
import 'package:dart_ml/src/di/injector.dart';
import 'package:di/di.dart';
import 'package:simd_vector/vector.dart';

class RegressorImpl implements Predictor {
  PredictorBase _predictor;

  RegressorImpl(Module module) {
    coreInjector ??= new ModuleInjector([module]);
    _predictor = new PredictorBase();
  }

  Metric get metric => _predictor.metric;

  void train(
    List<Float32x4Vector> features,
    List<double> labels,
    {Float32x4Vector weights}
  ) => _predictor.train(features, labels, weights: weights);

  double test(
    List<Float32x4Vector> features,
    List<double> origLabels,
    {MetricType metric}
  ) => _predictor.test(features, origLabels, metric: metric);

  Float32x4Vector predict(List<Float32x4Vector> features) => _predictor.predict(features);
}
