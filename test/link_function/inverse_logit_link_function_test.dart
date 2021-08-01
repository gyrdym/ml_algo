import 'package:ml_algo/src/common/exception/logit_scores_matrix_dimension_exception.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/inverse_logit_link_function.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

import '../helpers.dart';

void main() {
  void testInverseLogitLinkFunction(
      LinkFunction inverseLogitLink, DType dtype) {
    group('InverseLogitLinkFunction ($dtype)', () {
      test('should translate positive scores to probabilities', () {
        final scores = Matrix.fromList([
          [1.0],
          [2.0],
          [3.0],
          [4.0],
        ], dtype: dtype);
        final probabilities = inverseLogitLink.link(scores);

        expect(
            probabilities,
            iterable2dAlmostEqualTo([
              [0.73105],
              [0.88079],
              [0.9525],
              [0.98201]
            ], 1e-4));
        expect(probabilities.dtype, dtype);
      });

      test('should translate negative scores to probabilities', () {
        final scores = Matrix.fromList([
          [-1.0],
          [-2.0],
          [-3.0],
          [-4.0],
        ], dtype: dtype);
        final probabilities = inverseLogitLink.link(scores);

        expect(
            probabilities,
            iterable2dAlmostEqualTo([
              [0.268],
              [0.119],
              [0.047],
              [0.017]
            ], 1e-2));
        expect(probabilities.dtype, dtype);
      });

      test('should return 1 for positive scores which are out of range', () {
        final scores = Matrix.fromList([
          [50000.0],
          [100000.0],
          [200.0],
          [1000.0],
          [10.0],
        ], dtype: dtype);
        final probabilities = inverseLogitLink.link(scores);

        expect(
            probabilities,
            equals([
              [1.0],
              [1.0],
              [1.0],
              [1.0],
              [1.0],
            ]));
        expect(probabilities.dtype, dtype);
      });

      test('should return 0 for negative scores which are out of range', () {
        final scores = Matrix.fromList([
          [-10.0],
          [-11.0],
          [-12.0],
          [-13.0],
        ], dtype: dtype);
        final probabilities = inverseLogitLink.link(scores);

        expect(
            probabilities,
            equals([
              [0.0],
              [0.0],
              [0.0],
              [0.0]
            ]));
        expect(probabilities.dtype, dtype);
      });

      test(
          'should translate mixed collection of positive and negative scores '
          'to probabilities', () {
        final scores = Matrix.fromList([
          [1.0],
          [-2.0],
          [3.0],
          [-4.0],
        ], dtype: dtype);
        final probabilities = inverseLogitLink.link(scores);

        expect(
            probabilities,
            iterable2dAlmostEqualTo([
              [0.731],
              [0.119],
              [0.952],
              [0.017]
            ], 1e-3));
        expect(probabilities.dtype, dtype);
      });

      test('should translate zero scores to probabilities', () {
        final scores = Matrix.fromList([
          [0.0],
          [0.0],
          [0.0],
          [0.0],
        ], dtype: dtype);
        final probabilities = inverseLogitLink.link(scores);

        expect(
            probabilities,
            equals([
              [0.5],
              [0.5],
              [0.5],
              [0.5]
            ]));
        expect(probabilities.dtype, dtype);
      });

      test(
          'should throw an exception if scores matrix has more than 1 '
          'column', () {
        final scores = Matrix.fromList([
          [1.0, 10],
          [0.0, 200],
          [0.0, -100],
          [9.0, 1e7],
        ], dtype: dtype);
        final actual = () => inverseLogitLink.link(scores);

        expect(actual, throwsA(isA<LogitScoresMatrixDimensionException>()));
      });
    });
  }

  testInverseLogitLinkFunction(const InverseLogitLinkFunction(), DType.float32);
  testInverseLogitLinkFunction(const InverseLogitLinkFunction(), DType.float64);
}
