import 'package:ml_algo/src/metric/type.dart';
import 'package:ml_linalg/linalg.dart';

abstract class Evaluable<E, S extends MLVector<E>> {
  void fit(MLMatrix<E, MLVector<E>> features, MLVector<E> origLabels,
      {MLVector<E> initialWeights, bool isDataNormalized});

  double test(MLMatrix<E, MLVector<E>> features, MLVector<E> origLabels, MetricType metric);
}
