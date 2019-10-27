import 'package:ml_linalg/vector.dart';

void validateInitialCoefficients(Vector coefficients, bool fitIntercept,
    int featuresNumber) {

  final expectedNumber = fitIntercept
      ? featuresNumber + 1
      : featuresNumber;

  if (coefficients.length != expectedNumber) {
    throw Exception('Wrong initial coefficients vector provided: expected '
        'length ${expectedNumber}, but ${coefficients.length} given.');
  }
}
