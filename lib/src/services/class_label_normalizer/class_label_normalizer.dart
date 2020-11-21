import 'package:ml_linalg/matrix.dart';

abstract class ClassLabelNormalizer {
  Matrix normalize(Matrix labels, num positiveLabel,
      num negativeLabel);
}
