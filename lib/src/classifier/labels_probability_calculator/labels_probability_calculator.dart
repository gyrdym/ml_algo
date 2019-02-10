import 'package:ml_linalg/vector.dart';

abstract class LabelsProbabilityCalculator {
  MLVector getProbabilities(MLVector scores);
}
