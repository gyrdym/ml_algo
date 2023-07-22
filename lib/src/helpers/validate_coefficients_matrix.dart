import 'package:ml_linalg/matrix.dart';

/// Parameters:
///
/// [coefficients] A `N * K` matrix, where `N` - number of coefficients, `K` -
/// number of targets (in case of classification - number of classes, in case of
/// regression this value is always 1)
///
/// [featuresNum] Number of features. The number of rows of [coefficients]
/// matrix should be equal to [featuresNum]
void validateCoefficientsMatrix(Matrix coefficients, [num? featuresNum]) {
  if (!coefficients.hasData) {
    throw Exception('No coefficients provided');
  }

  if (featuresNum != null && coefficients.rowCount != featuresNum) {
    throw Exception('Wrong features number provided: expected '
        '${coefficients.rowCount}, but $featuresNum given. '
        'Please, recheck columns number of the passed feature matrix');
  }
}
