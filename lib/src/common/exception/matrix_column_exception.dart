class MatrixColumnException implements Exception {
  MatrixColumnException(int rowsCount, int columnsCount) :
        message = 'Expected a matrix column, but matrix of shape '
            '($rowsCount, $columnsCount) given';

  final String message;
}
