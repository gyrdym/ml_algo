import 'dart:typed_data';

import 'package:ml_algo/src/score_to_prob_mapper/logit/logit_mapper.dart';
import 'package:ml_algo/src/score_to_prob_mapper/softmax/softmax_mapper.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

import '../test_utils/helpers/floating_point_iterable_matchers.dart';

void main() {
  group('LogitMapper', () {
    test('should translate scores to probabilities for Float32x4', () {
      final scores = Matrix.from([[1.0, 2.0, 3.0, 4.0]]);
      final logitLink = LogitMapper(Float32x4);
      final probabilities = logitLink.linkScoresToProbs(scores);

      expect(probabilities,
          matrixAlmostEqualTo([[0.73105, 0.88079, 0.9525, 0.98201]], 1e-4));
    });
  });

  group('SoftmaxMapper', () {
    test('should translate scores to probabilities for Float32x4', () {
      final logitLink = SoftmaxMapper(Float32x4);

      final scores = Matrix.columns([
        Vector.from([10.0, 55.0, 33.0, 29.0, 66.0]),
        Vector.from([20.0, 32.0, 44.0, 89.0, 41.0]),
        Vector.from([30.0, 21.0, 77.0, 40.0, 99.0]),
      ]);

      final probabilities = logitLink.linkScoresToProbs(scores);
      final expected = [
        [2.061060046209062e-9, 0.00004539786860886666, 0.9999546000703311],
        [0.9999999998973794, 1.0261879630648809e-10, 1.7139084313661305e-15],
        [7.78113224113376e-20, 4.658886145103376e-15, 0.9999999999999954,],
        [8.756510762696521e-27, 1.0, 5.242885663363464e-22],
        [4.6588861451033764e-15, 6.47023492564543e-26, 0.9999999999999953],
      ];
      expect(probabilities, matrixAlmostEqualTo(expected, 1e-4));
    });
  });
}
