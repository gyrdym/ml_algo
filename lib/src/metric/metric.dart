import 'package:linalg/vector.dart';

abstract class Metric {
  double getError(Vector predictedLabels, Vector origLabels);
}