import 'package:ml_algo/src/common/exception/matrix_column_exception.dart';
import 'package:ml_linalg/matrix.dart';

Matrix binarizeColumnMatrix(Matrix source) {
  if (source.columnsNum != 1) {
    throw MatrixColumnException(source.rowsNum, source.columnsNum);
  }

  final sourceAsVector = source
      .toVector();
  final binarizedVectors = sourceAsVector
      .unique()
      .map(
          (targetValue) => sourceAsVector
              .mapToVector(
                  (sourceValue) => sourceValue == targetValue
                      ? 1 : 0)).toList();

  return Matrix.fromColumns(binarizedVectors, dtype: source.dtype);
}
