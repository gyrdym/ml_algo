import 'package:ml_linalg/matrix.dart';

Matrix normalizeClassLabels(
        Matrix labels, num positiveLabel, num negativeLabel) =>
    positiveLabel != 1 || negativeLabel != 0
        ? labels.mapElements((label) => label == positiveLabel ? 1 : 0)
        : labels;
