import 'package:ml_algo/src/services/class_label_normalizer/class_label_normalizer.dart';
import 'package:ml_linalg/matrix.dart';

class ClassLabelNormalizerImpl implements ClassLabelNormalizer {
  const ClassLabelNormalizerImpl();

  @override
  Matrix normalize(Matrix labels, num positiveLabel,
      num negativeLabel) => positiveLabel != 1 || negativeLabel != 0
      ? labels.mapElements((label) => label == positiveLabel ? 1.0 : 0.0)
      : labels;
}
