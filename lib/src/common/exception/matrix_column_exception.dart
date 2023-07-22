import 'package:ml_linalg/matrix.dart';

class MatrixColumnException implements Exception {
  MatrixColumnException(Matrix matrix)
      : message = 'Expected a matrix column, matrix of shape '
            '(${matrix.rowCount}, ${matrix.columnCount}) given';

  final String message;

  @override
  String toString() => message;
}
