import 'dart:typed_data';

import 'package:ml_algo/src/score_to_prob_mapper/logit_mapper.dart';
import 'package:ml_algo/src/score_to_prob_mapper/softmax_mapper.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

import '../test_utils/helpers/floating_point_iterable_matchers.dart';

void main() {
  group('LogitMapper', () {
    test('should translate scores to probabilities for Float32x4', () {
      final scores = MLVector.from([1.0, 2.0, 3.0, 4.0]);
      final logitLink = LogitMapper(Float32x4);
      final probabilities = logitLink.linkScoresToProbs(scores);

      expect(probabilities,
          vectorAlmostEqualTo([0.73105, 0.88079, 0.9525, 0.98201], 1e-4));
    });
  });

  group('SoftmaxMapper', () {
    test('should translate scores to probabilities for Float32x4', () {
      final scores = MLVector.from([10.0, 55.0, 33.0, 29.0, 66.0]);
      final scoresByClasses = MLMatrix.from([
        //1st  2nd   3rd
        [10.0, 20.0, 30.0],
        [55.0, 32.0, 21.0],
        [33.0, 44.0, 77.0],
        [29.0, 89.0, 40.0],
        [66.0, 41.0, 99.0],
      ]);
      final logitLink = SoftmaxMapper(Float32x4);
      final probabilities = logitLink.linkScoresToProbs(scores,
          scoresByClasses);
      final expected = [2.061060046209062e-9, 0.9999999998973794,
      7.78113224113376e-20, 8.756510762696521e-27, 4.6588861451033764e-15];

      expect(probabilities,
          vectorAlmostEqualTo(expected, 1e-4));
    });
  });
}
