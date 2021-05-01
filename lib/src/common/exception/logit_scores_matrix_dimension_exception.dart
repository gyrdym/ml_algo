class LogitScoresMatrixDimensionException implements Exception {
  LogitScoresMatrixDimensionException(int columnsCount)
      : message = 'Logit link function: wrong scores matrix dimension, '
            'expected columns count 1, $columnsCount given';

  final String message;

  @override
  String toString() => message;
}
