import 'package:ml_linalg/matrix.dart';

abstract class Metric {
  double getScore(MLMatrix predictedLabels, MLMatrix origLabels);
}
