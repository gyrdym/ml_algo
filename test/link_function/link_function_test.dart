import 'package:ml_algo/src/link_function/logit/inverse_logit_link_function.dart';
import 'package:ml_algo/src/link_function/softmax/softmax_link_function.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:ml_tech/unit_testing/matchers/iterable_2d_almost_equal_to.dart';
import 'package:test/test.dart';

void main() {
  group('InverseLogitLinkFunction', () {
    test('should translate scores to probabilities for Float32x4', () {
      final scores = Matrix.fromList([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
      ]);
      final inverseLogitLink = InverseLogitLinkFunction(DType.float32);
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
      final inverseLogitLink = InverseLogitLinkFunction(DType.float32);
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
      final inverseLogitLink = InverseLogitLinkFunction(DType.float32);
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
      final inverseLogitLink = InverseLogitLinkFunction(DType.float32);
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

  group('SoftmaxLinkFunction', () {
    test('should translate scores to probabilities for Float32x4', () {
      final logitLink = SoftmaxLinkFunction(DType.float32);

      final scores = Matrix.fromColumns([
        Vector.fromList([10.0, 55.0, 33.0, 29.0, 66.0]),
        Vector.fromList([20.0, 32.0, 44.0, 89.0, 41.0]),
        Vector.fromList([30.0, 21.0, 77.0, 40.0, 99.0]),
      ]);

      final probabilities = logitLink.link(scores);
      final expected = [
        [2.061060046209062e-9, 0.00004539786860886666, 0.9999546000703311],
        [0.9999999998973794, 1.0261879630648809e-10, 1.7139084313661305e-15],
        [7.78113224113376e-20, 4.658886145103376e-15, 0.9999999999999954,],
        [8.756510762696521e-27, 1.0, 5.242885663363464e-22],
        [4.6588861451033764e-15, 6.47023492564543e-26, 0.9999999999999953],
      ];
      expect(probabilities, iterable2dAlmostEqualTo(expected, 1e-4));
    });
  });
}
