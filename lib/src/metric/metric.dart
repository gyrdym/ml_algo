import 'package:ml_linalg/vector.dart';

abstract class Metric {
  double getScore(MLVector predictedLabels, MLVector origLabels);
}
