import 'package:ml_algo/src/metric/type.dart';
import 'package:ml_linalg/linalg.dart';

abstract class Evaluable<E> {
  void fit(MLMatrix<E> features, MLVector<E> origLabels, {MLVector<E> initialWeights, bool isDataNormalized});
  double test(MLMatrix<E> features, MLVector<E> origLabels, MetricType metric);
}
