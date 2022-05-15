import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/helpers/get_binary_representation.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('getBinaryRepresentation', () {
    test('should return correct representation, digitCapacity=4, seed=10', () {
      final data = Matrix.fromList([
        [23, 12, 34],
        [16, 1, 7],
        [-19, 2, -109],
        [-23, -12, 93],
        [101, -10, -34],
        [1, 10, 11],
      ]);
      final randomMatrix = Matrix.fromList([
        [
          983.0911254882812,
          127.48931121826172,
          833.928955078125,
          893.8758544921875
        ],
        [
          667.7393798828125,
          -63.989959716796875,
          592.6019897460938,
          -92.79692840576172
        ],
        [
          -754.34326171875,
          -248.08624267578125,
          -275.7384033203125,
          279.0705871582031
        ],
      ]);
      final actual = getBinaryRepresentation(data, randomMatrix);

      expect(actual, [
        [1, 0, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 1, 1],
        [0, 0, 1, 1],
      ]);
    });
  });
}
