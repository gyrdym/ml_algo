import 'package:ml_linalg/matrix.dart';

void validateCoefficientsMatrix(Matrix coefficients, [num expectedColumnsNum]) {
  if (!coefficients.hasData) {
    throw Exception('No coefficients provided');
  }

  if (expectedColumnsNum != null && coefficients.rowsNum != expectedColumnsNum) {
    throw Exception('Wrong features number provided: expected '
        '${coefficients.rowsNum}, but ${expectedColumnsNum} given. '
        'Please, recheck columns number of the passed feature matrix');
  }
}
