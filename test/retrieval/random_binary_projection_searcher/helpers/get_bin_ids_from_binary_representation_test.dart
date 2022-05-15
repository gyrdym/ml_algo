import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/helpers/get_indices_from_binary_representation.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('getBinIdsFromBinaryRepresentation', () {
    test('should convert binary values to decimal ones', () {
      final data = Matrix.fromList([
        [1, 0, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 1, 1],
        [0, 0, 1, 1],
      ]);
      final actual = getBinIdsFromBinaryRepresentation(data);

      expect(actual, [13, 15, 7, 8, 15, 12]);
    });
  });
}
