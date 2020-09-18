import 'package:ml_algo/src/common/exception/matrix_column_exception.dart';
import 'package:ml_linalg/matrix.dart';

void validateMatrixColumns(Iterable<Matrix> matrices) {
  final firstInvalidMatrix = matrices
      .firstWhere((matrix) => matrix.columnsNum != 1, orElse: () => null);

  if (firstInvalidMatrix == null) {
    return;
  }

  throw MatrixColumnException(firstInvalidMatrix);
}
