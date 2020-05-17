import 'package:ml_algo/src/link_function/logit/float32_inverse_logit_function.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_tech/unit_testing/matchers/iterable_2d_almost_equal_to.dart';
import 'package:test/test.dart';

void main() {
  group('Float32InverseLogitLinkFunction', () {
    test('should translate scores to probabilities for Float32x4', () {
      final scores = Matrix.fromList([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
      ]);
      final inverseLogitLink = const Float32InverseLogitLinkFunction();
      final probabilities = inverseLogitLink.link(scores);

      expect(probabilities,
          iterable2dAlmostEqualTo([
            [0.73105],
            [0.88079],
            [0.9525],
            [0.98201]
          ], 1e-4));
    });

    test('should not yield NaN as probability values', () {
      final scores = Matrix.fromList([
        [50000.0],
        [100000.0],
        [200.0],
        [1000.0],
      ]);
      final inverseLogitLink = const Float32InverseLogitLinkFunction();
      final probabilities = inverseLogitLink.link(scores);
      final probaAsVector = probabilities.getColumn(0);

      expect(probaAsVector[0], isNotNaN);
      expect(probaAsVector[1], isNotNaN);
      expect(probaAsVector[2], isNotNaN);
      expect(probaAsVector[3], isNotNaN);

      expect(probabilities,
          equals([
            [1.0],
            [1.0],
            [1.0],
            [1.0]
          ]));
    });

    test('should not yield extremelly precise numbers that are close to one '
        'as probability values', () {
      final scores = Matrix.fromList([
        [10.0],
        [11.0],
        [12.0],
        [13.0],
      ]);
      final inverseLogitLink = const Float32InverseLogitLinkFunction();
      final probabilities = inverseLogitLink.link(scores);
      final probaAsVector = probabilities.getColumn(0);

      expect(probaAsVector[0], isNotNaN);
      expect(probaAsVector[1], isNotNaN);
      expect(probaAsVector[2], isNotNaN);
      expect(probaAsVector[3], isNotNaN);

      expect(probabilities,
          equals([
            [1.0],
            [1.0],
            [1.0],
            [1.0]
          ]));
    });

    test('should not yield extremelly precise numbers that are close to zero '
        'as probability values', () {
      final scores = Matrix.fromList([
        [-10.0],
        [-11.0],
        [-12.0],
        [-13.0],
      ]);
      final inverseLogitLink = const Float32InverseLogitLinkFunction();
      final probabilities = inverseLogitLink.link(scores);
      final probaAsVector = probabilities.getColumn(0);

      expect(probaAsVector[0], isNotNaN);
      expect(probaAsVector[1], isNotNaN);
      expect(probaAsVector[2], isNotNaN);
      expect(probaAsVector[3], isNotNaN);

      expect(probabilities,
          equals([
            [0.0],
            [0.0],
            [0.0],
            [0.0]
          ]));
    });
  });
}
