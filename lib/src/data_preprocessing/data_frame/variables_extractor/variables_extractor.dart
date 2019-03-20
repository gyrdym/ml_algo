import 'package:ml_linalg/matrix.dart';

abstract class VariablesExtractor {
  Matrix extractFeatures();
  Matrix extractLabels();
}
