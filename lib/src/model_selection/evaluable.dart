import 'package:dart_ml/src/metric/type.dart';
import 'package:linalg/vector.dart';

abstract class Evaluable<E, S extends List<E>, T extends List<double>> {
  void fit(
    List<SIMDVector<S, T, E>> features,
    SIMDVector<S, T, E> origLabels,
    {
      SIMDVector<S, T, E> initialWeights,
      bool isDataNormalized
    });

  double test(
    List<SIMDVector<S, T, E>> features,
    SIMDVector<S, T, E> origLabels,
    MetricType metric
  );
}
