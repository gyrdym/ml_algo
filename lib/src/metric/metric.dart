import 'package:ml_linalg/vector.dart';

abstract class Metric {
  double getError(MLVector predictedLabels, MLVector origLabels);
}
