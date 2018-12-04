import 'package:ml_linalg/linalg.dart';

abstract class Metric<E> {
  double getError(MLVector<E> predictedLabels, MLVector<E> origLabels);
}
