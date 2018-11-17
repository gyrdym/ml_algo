import 'package:linalg/vector.dart';

abstract class Metric<E> {
  double getError(Vector<E> predictedLabels, Vector<E> origLabels);
}
