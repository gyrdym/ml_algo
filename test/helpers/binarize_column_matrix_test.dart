import 'package:ml_algo/src/common/exception/matrix_column_exception.dart';
import 'package:ml_algo/src/helpers/binarize_column_matrix.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('binarizeColumnMatrix', () {
    final classLabel1 = 21.0;
    final classLabel2 = -1001.0;
    final classLabel3 = 0.0;
    final classLabel4 = 0.67;
    final source = [
      [classLabel2],
      [classLabel1],
      [classLabel1],
      [classLabel3],
      [classLabel2],
      [classLabel3],
      [classLabel1],
    ];
    final sourceFloat32Matrix = Matrix.fromList(source, dtype: DType.float32);
    final sourceFloat64Matrix = Matrix.fromList(source, dtype: DType.float64);
    final binarizedMatrix = Matrix.fromList([
      [1, 0, 0],
      [0, 1, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 0, 0],
      [0, 0, 1],
      [0, 1, 0],
    ]);
    final distinctMatrix = Matrix.fromList([
      [classLabel2],
      [classLabel1],
      [classLabel4],
      [classLabel3],
    ]);
    final binarizedDistinctMatrix = Matrix.fromList([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ]);

    test('should throw an exception if a matrix of improper shape was '
        'passed', () {
      final incorrectMatrix = Matrix.fromList([
        [21, 34, 33],
      ]);

      expect(() => binarizeColumnMatrix(incorrectMatrix),
          throwsA(isA<MatrixColumnException>()));
    });

    test('should throw an exception if an empty matrix was passed', () {
      final incorrectMatrix = Matrix.fromList([]);

      expect(() => binarizeColumnMatrix(incorrectMatrix),
          throwsA(isA<MatrixColumnException>()));
    });

    test('should handle float32 matrix', () {
      expect(binarizeColumnMatrix(sourceFloat32Matrix).dtype,
          sourceFloat32Matrix.dtype);
    });

    test('should handle float64 matrix', () {
      expect(binarizeColumnMatrix(sourceFloat64Matrix).dtype,
          sourceFloat64Matrix.dtype);
    });

    test('should binarize source matrix', () {
      expect(binarizeColumnMatrix(sourceFloat32Matrix), binarizedMatrix);
    });

    test('should binarize matrix with all distinct elements', () {
      expect(binarizeColumnMatrix(distinctMatrix), binarizedDistinctMatrix);
    });
  });
}
