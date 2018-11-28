import 'package:linalg/linalg.dart';

abstract class Metric<E> {
  double getError(Vector<E> predictedLabels, Vector<E> origLabels);
}
