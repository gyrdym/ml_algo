import 'package:dart_ml/src/metric/metric.dart';
import 'package:linalg/linalg.dart';

abstract class ClassificationMetric<E> implements Metric<E> {
  @override
  double getError(Vector<E> predictedLabels, Vector<E> origLabels);
  double getScore(Vector<E> predictedLabels, Vector<E> origLabels);
}
