void validateTreeSolverMinError(num minError) {
  if (minError < 0 || minError > 1) {
    throw RangeError.range(minError, 0, 1, 'wrong minimal error value');
  }
}
