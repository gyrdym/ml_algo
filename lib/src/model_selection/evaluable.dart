import 'package:dart_ml/src/metric/type.dart';
import 'package:linalg/linalg.dart';

abstract class Evaluable<E> {
  void fit(List<Vector<E>> features, Vector<E> origLabels, {
      Vector<E> initialWeights,
      bool isDataNormalized
    });

  double test(List<Vector<E>> features, Vector<E> origLabels, MetricType metric);
}
