import 'dart:typed_data';

import 'package:ml_algo/src/link_function/logit_link_function.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

import '../test_utils/helpers/floating_point_iterable_matchers.dart';

void main() {
  group('Float32x4 logit link function', () {
    test('should properly translate score to probability', () {
      final scores = MLVector.from([1.0, 2.0, 3.0, 4.0]);
      final logitLink = LogitLinkFunction(Float32x4);
      final probabilities = logitLink.linkScoresToProbs(scores);

      expect(probabilities,
          vectorAlmostEqualTo([0.73105, 0.88079, 0.9525, 0.98201], 1e-4));
    });
  });
}
