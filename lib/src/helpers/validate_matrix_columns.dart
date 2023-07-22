import 'package:ml_algo/src/common/exception/matrix_column_exception.dart';
import 'package:ml_linalg/matrix.dart';

void validateMatrixColumns(Iterable<Matrix> matrices) {
  final matrixStub = Matrix.empty();
  final firstInvalidMatrix = matrices.firstWhere(
      (matrix) => matrix.columnCount != 1,
      orElse: () => matrixStub);

  if (identical(firstInvalidMatrix, matrixStub)) {
    return;
  }

  throw MatrixColumnException(firstInvalidMatrix);
}
