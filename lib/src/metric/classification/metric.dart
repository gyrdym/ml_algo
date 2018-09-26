import 'package:dart_ml/src/metric/metric.dart';
import 'package:linalg/vector.dart';

abstract class ClassificationMetric<S extends List<E>, T extends List<double>, E>
    implements Metric<S, T, E> {

  @override
  double getError(SIMDVector<S, T, E> predictedLabels, SIMDVector<S, T, E> origLabels);
  double getScore(SIMDVector<S, T, E> predictedLabels, SIMDVector<S, T, E> origLabels);
}
