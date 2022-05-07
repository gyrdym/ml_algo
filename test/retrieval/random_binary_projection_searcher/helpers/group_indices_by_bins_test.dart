import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/helpers/group_indices_by_bins.dart';
import 'package:test/test.dart';

void main() {
  group('groupIndicesByBins', () {
    test('should return grouped indices', () {
      final binIds = [13, 15, 7, 8, 15, 12];
      final actual = groupIndicesByBins(binIds);
      final expected = {
        7: [2],
        8: [3],
        12: [5],
        13: [0],
        15: [1, 4]
      };

      expect(actual, expected);
    });
  });
}
