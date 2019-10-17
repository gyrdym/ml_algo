import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_impl.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

void main() {
  group('KnnClassifierImpl', () {
    test('should throw an exception if empty matrix is provided as feature '
        'matrix', () {
      final actual = () => KnnClassifierImpl(
        Matrix.empty(dtype: DType.float32),
        Matrix.fromList([[1, 1, 1]]),
        'target',
            ($, [$$]) => null,
        2,
        Distance.cosine,
            ($, $$, $$$, $$$$, {distance, standardize}) => null,
        DType.float32,
      );

      expect(actual, throwsException);
    });

    test('should throw an exception if empty matrix is provided as outcome '
        'matrix', () {
      final actual = () => KnnClassifierImpl(
        Matrix.fromList([[1, 1, 1]]),
        Matrix.empty(dtype: DType.float32),
        'target',
            ($, [$$]) => null,
        2,
        Distance.cosine,
            ($, $$, $$$, $$$$, {distance, standardize}) => null,
        DType.float32,
      );

      expect(actual, throwsException);
    });

    test('should throw an exception if columns numbers of feature and outcome '
        'matrices do not match', () {
      final actual = () => KnnClassifierImpl(
        Matrix.fromList([[1, 1, 1]]),
        Matrix.fromList([[1, 1, 1, 1]]),
        'target',
            ($, [$$]) => null,
        2,
        Distance.cosine,
            ($, $$, $$$, $$$$, {distance, standardize}) => null,
        DType.float32,
      );

      expect(actual, throwsException);
    });

    test('should throw an exception if k parameter is greater than the number '
        'of rows of provided matrices', () {
      final actual = () => KnnClassifierImpl(
        Matrix.fromList([[1, 1, 1, 1]]),
        Matrix.fromList([[1, 1, 1, 1]]),
        'target',
            ($, [$$]) => null,
        2,
        Distance.cosine,
            ($, $$, $$$, $$$$, {distance, standardize}) => null,
        DType.float32,
      );

      expect(actual, throwsException);
    });
  });
}
