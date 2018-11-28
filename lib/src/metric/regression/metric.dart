import 'package:dart_ml/src/metric/metric.dart';
import 'package:linalg/linalg.dart';

abstract class RegressionMetric<E> implements Metric<E> {
  @override
  double getError(Vector<E> predictedLabels, Vector<E> origLabels);
}
