import 'package:dart_ml/src/metric/type.dart';
import 'package:linalg/linalg.dart';

abstract class Evaluable<E, S extends Vector<E>> {
  void fit(Matrix<E, Vector<E>> features, Vector<E> origLabels, {
      Vector<E> initialWeights,
      bool isDataNormalized
    });

  double test(Matrix<E, Vector<E>> features, Vector<E> origLabels, MetricType metric);
}
