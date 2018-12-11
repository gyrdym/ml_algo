import 'dart:typed_data';

import 'package:ml_algo/src/score_to_prob_link_function/link_function_impl.dart';
import 'package:test/test.dart';

void main() {
  group('Vectorized logit link function', () {
    test('should properly translate score to probability', () {
      final scores = Float32x4(1.0, 2.0, 3.0, 4.0);
      final probabilities = vectorizedLogitLink(scores);

      expect(probabilities.x, inInclusiveRange(0.731, 0.732));
      expect(probabilities.y, inInclusiveRange(0.88, 0.881));
      expect(probabilities.z, inInclusiveRange(0.952, 0.953));
      expect(probabilities.w, inInclusiveRange(0.982, 0.983));
    });
  });
}
