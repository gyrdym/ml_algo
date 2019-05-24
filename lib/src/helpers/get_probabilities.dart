import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/matrix.dart';

Matrix getProbabilities(Matrix features, Matrix coefficients,
    LinkFunction linkFunction) {
  if (features.columnsNum != coefficients.rowsNum) {
    throw Exception('Wrong features number provided: expected '
        '${coefficients.rowsNum}, but ${features.columnsNum} given. '
        'Please, recheck columns number of the passed feature matrix');
  }
  return linkFunction.link(features * coefficients);
}
