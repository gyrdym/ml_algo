import 'package:ml_linalg/matrix.dart';

abstract class Metric {
  double getScore(Matrix predictedLabels, Matrix origLabels);
}
