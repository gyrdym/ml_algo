import 'package:ml_linalg/matrix.dart';

Matrix normalizeClassLabels(Matrix labels, num positiveLabel,
    num negativeLabel) => positiveLabel != 1 || negativeLabel != 0
      ? labels.mapRows(
          (row) => row.mapToVector(
              (label) => label == positiveLabel ? 1.0 : 0.0))
      : labels;
