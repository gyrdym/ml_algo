import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/softmax/float32_softmax_link_function.dart';
import 'package:ml_algo/src/link_function/softmax/float64_softmax_link_function.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

import '../../helpers.dart';

void main() {
  void testSoftmaxLinkFunction(LinkFunction linkFunction, DType dtype) {
    group(linkFunction.runtimeType, () {
      test('should translate positive scores to probabilities', () {
        final scores = Matrix.fromList([
          [ 2,  1, -3],
          [ 7, 11,  0],
          [-7,  4, -9],
          [ 6,  1,  3],
          [-1, -3, -2],
        ], dtype: dtype);
        final probabilities = linkFunction.link(scores);
        final expected = [
          [0.727,   0.267, 0.004   ],
          [0.017,   0.981, 0.00001 ],
          [0.00001, 0.999, 0.000002],
          [0.946,   0.006, 0.047   ],
          [0.665,   0.09,  0.244   ],
        ];

        expect(probabilities, iterable2dAlmostEqualTo(expected, 1e-3));
        expect(probabilities.dtype, dtype);
      });
    });
  }

  testSoftmaxLinkFunction(const Float32SoftmaxLinkFunction(), DType.float32);
  testSoftmaxLinkFunction(const Float64SoftmaxLinkFunction(), DType.float64);
}
