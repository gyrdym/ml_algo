import 'package:ml_algo/src/metric/type.dart';
import 'package:ml_linalg/linalg.dart';

abstract class Predictor<E> {
  /// Fits the given data ([features]) to true labels ([origLabels]). It's possible to provide [initialWeights]
  /// and specify, whether the [features] normalized or not
  void fit(MLMatrix<E> features, MLVector<E> origLabels, {MLVector<E> initialWeights, bool isDataNormalized});

  /// Assesses model according to provided [metric]
  double test(MLMatrix<E> features, MLVector<E> origLabels, MetricType metric);
}
