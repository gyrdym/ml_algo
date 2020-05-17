import 'package:ml_algo/src/link_function/softmax/float32_softmax_link_function.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:ml_tech/unit_testing/matchers/iterable_2d_almost_equal_to.dart';
import 'package:test/test.dart';

void main() {
  group('Float32SoftmaxLinkFunction', () {
    test('should translate scores to probabilities for Float32x4', () {
      final logitLink = const Float32SoftmaxLinkFunction();

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
