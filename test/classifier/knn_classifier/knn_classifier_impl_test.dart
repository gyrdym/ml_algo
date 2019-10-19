import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

void main() {
  group('KnnClassifierImpl', () {
    group('constructor', () {
      test('should throw an exception if empty matrix is provided as feature '
          'matrix', () {
        final actual = () => KnnClassifierImpl(
          Matrix.empty(dtype: DType.float32),
          Matrix.fromList([[1]]),
          'target',
              (_, [__]) => null,
          2,
          Distance.cosine,
              (_, __, ___, ____, {distance, standardize}) => null,
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
              (_, [__]) => null,
          1,
          Distance.cosine,
              (_, __, ___, ____, {distance, standardize}) => null,
          DType.float32,
        );

        expect(actual, throwsException);
      });

      test('should throw an exception if rows number of outcome '
          'matrix is greater than the rows number of feature matrix', () {
        final featureMatrix = Matrix.fromList([[1, 1, 1]]);
        final outcomeMatrix = Matrix.fromList([
          [1],
          [2],
        ]);

        final actual = () => KnnClassifierImpl(
          featureMatrix,
          outcomeMatrix,
          'target',
              (_, [__]) => null,
          1,
          Distance.cosine,
              (_, __, ___, ____, {distance, standardize}) => null,
          DType.float32,
        );

        expect(actual, throwsException);
      });

      test('should throw an exception if rows number of outcome '
          'matrix is less than the rows number of feature matrix', () {
        final featureMatrix = Matrix.fromList([
          [1, 1, 1],
          [2, 2, 2],
        ]);
        final outcomeMatrix = Matrix.fromList([
          [1],
        ]);

        final actual = () => KnnClassifierImpl(
          featureMatrix,
          outcomeMatrix,
          'target',
              (_, [__]) => null,
          1,
          Distance.cosine,
              (_, __, ___, ____, {distance, standardize}) => null,
          DType.float32,
        );

        expect(actual, throwsException);
      });

      test('should throw an exception if outcome matrix is not a column '
          'matrix', () {
        final featureMatrix = Matrix.fromList([
          [1, 1, 1],
          [2, 2, 2],
        ]);
        final outcomeMatrix = Matrix.fromList([
          [1, 1],
          [2, 2],
        ]);

        final actual = () => KnnClassifierImpl(
          featureMatrix,
          outcomeMatrix,
          'target',
              (_, [__]) => null,
          1,
          Distance.cosine,
              (_, __, ___, ____, {distance, standardize}) => null,
          DType.float32,
        );

        expect(actual, throwsException);
      });

      test('should throw an exception if k parameter is greater than the number '
          'of rows of provided matrices', () {
        final actual = () => KnnClassifierImpl(
          Matrix.fromList([[1, 1, 1, 1]]),
          Matrix.fromList([[1]]),
          'target',
              (_, [__]) => null,
          2,
          Distance.cosine,
              (_, __, ___, ____, {distance, standardize}) => null,
          DType.float32,
        );

        expect(actual, throwsRangeError);
      });

      test('should throw an exception if k parameter is less than 0', () {
        final actual = () => KnnClassifierImpl(
          Matrix.fromList([[1, 1, 1, 1]]),
          Matrix.fromList([[1]]),
          'target',
              (_, [__]) => null,
          -1,
          Distance.cosine,
              (_, __, ___, ____, {distance, standardize}) => null,
          DType.float32,
        );

        expect(actual, throwsRangeError);
      });

      test('should throw an exception if k parameter is equal to 0', () {
        final actual = () => KnnClassifierImpl(
          Matrix.fromList([[1, 1, 1, 1]]),
          Matrix.fromList([[1]]),
          'target',
              (_, [__]) => null,
          0,
          Distance.cosine,
              (_, __, ___, ____, {distance, standardize}) => null,
          DType.float32,
        );

        expect(actual, throwsRangeError);
      });
    });

    group('predict method', () {
      test('should throw an exception if no features are provided', () {
        final classifier = KnnClassifierImpl(
          Matrix.fromList([[1, 1, 1, 1]]),
          Matrix.fromList([[1]]),
          'target',
          (_, [__]) => null,
          1,
          Distance.cosine,
          (_, __, ___, ____, {distance, standardize}) => null,
          DType.float32,
        );

        final features = DataFrame.fromMatrix(Matrix.empty());

        expect(() => classifier.predict(features), throwsException);
      });

      test('should throw an exception if the number of a provided feature '
          'matrix columns is less than the number of columns of train '
          'feature matrix', () {
        final classifier = KnnClassifierImpl(
          Matrix.fromList([
            [1, 1, 1, 1],
            [2, 2, 2, 2],
          ]),
          Matrix.fromList([
            [1],
            [3],
          ]),
          'target',
          (_, [__]) => null,
          1,
          Distance.cosine,
          (_, __, ___, ____, {distance, standardize}) => null,
          DType.float32,
        );

        final features = DataFrame.fromMatrix(Matrix.fromList(
          [
            [10, 10, 10],
          ],
          dtype: DType.float32,
        ));

        expect(() => classifier.predict(features), throwsException);
      });

      test('should throw an exception if the number of a provided feature '
          'matrix columns is greater than the number of columns of train '
          'feature matrix', () {
        final classifier = KnnClassifierImpl(
          Matrix.fromList([
            [1, 1, 1, 1],
            [2, 2, 2, 2],
          ]),
          Matrix.fromList([
            [1],
            [3],
          ]),
          'target',
              (_, [__]) => null,
          1,
          Distance.cosine,
              (_, __, ___, ____, {distance, standardize}) => null,
          DType.float32,
        );

        final features = DataFrame.fromMatrix(Matrix.fromList(
          [
            [10, 10, 10, 100, 100],
          ],
          dtype: DType.float32,
        ));

        expect(() => classifier.predict(features), throwsException);
      });
    });
  });
}
