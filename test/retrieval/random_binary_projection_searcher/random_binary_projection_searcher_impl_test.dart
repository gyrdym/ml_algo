import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/random_binary_projection_searcher_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() {
  group('RandomBinaryProjectionSearcherImpl', () {
    final data = DataFrame([
      [23, 12, 34],
      [16, 1, 7],
      [-19, 2, -109],
      [-23, -12, 93],
      [101, -10, -34],
      [1, 10, 11],
    ], headerExists: false);
    final digitCapacity = 4;
    final searcher = RandomBinaryProjectionSearcherImpl(
        data.header, data.toMatrix(), digitCapacity,
        seed: 10);

    test('should build bin map', () {
      expect(searcher.bins, {
        7: [2],
        8: [3],
        12: [5],
        13: [0],
        15: [1, 4]
      });
    });

    test('should persist random vectors', () {
      expect(searcher.randomVectors, [
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
    });

    test('should persist digit capacity', () {
      expect(searcher.digitCapacity, digitCapacity);
    });

    test('should persist header', () {
      expect(searcher.columns, data.header);
    });

    test('should perform knn search, k=4', () {
      final k = 4;
      final searchRadius = 3;
      final result =
          searcher.query(Vector.fromList([-19, 2, -109]), k, searchRadius);

      expect(result, hasLength(k));
      expect(result.elementAt(0).index, 2);
      expect(result.elementAt(0).distance, 0);
    });

    test('should perform knn search, k=3', () {
      final k = 3;
      final searchRadius = 3;
      final result =
          searcher.query(Vector.fromList([1, 10, 11]), k, searchRadius);

      expect(result, hasLength(k));
      expect(result.elementAt(0).index, 5);
      expect(result.elementAt(0).distance, 0);
    });

    test('should perform knn search, k=2', () {
      final k = 2;
      final searchRadius = 3;
      final result =
          searcher.query(Vector.fromList([101, -10, -34]), k, searchRadius);

      expect(result, hasLength(k));
      expect(result.elementAt(0).index, 4);
      expect(result.elementAt(0).distance, 0);
    });

    test('should perform knn search, k=1', () {
      final k = 1;
      final searchRadius = 3;
      final result =
          searcher.query(Vector.fromList([23, 12, 34]), k, searchRadius);

      expect(result, hasLength(k));
      expect(result.elementAt(0).index, 0);
      expect(result.elementAt(0).distance, 0);
    });

    test('should perform knn search, k=3, searchRadius=2', () {
      final k = 3;
      final searchRadius = 2;
      final result =
          searcher.query(Vector.fromList([1, 10, 11]), k, searchRadius);

      expect(result, hasLength(k));
      expect(result.elementAt(0).index, 5);
      expect(result.elementAt(0).distance, 0);
    });

    test('should perform knn search, k=3, searchRadius=1', () {
      final k = 3;
      final searchRadius = 1;
      final result =
          searcher.query(Vector.fromList([1, 10, 11]), k, searchRadius);

      expect(result, hasLength(3));
      expect(result.elementAt(0).index, 5);
      expect(result.elementAt(0).distance, 0);
    });

    test('should perform knn search, k=3, searchRadius=0', () {
      final k = 3;
      final searchRadius = 0;
      final result =
          searcher.query(Vector.fromList([1, 10, 11]), k, searchRadius);

      expect(result, hasLength(1));
      expect(result.elementAt(0).index, 5);
      expect(result.elementAt(0).distance, 0);
    });

    test('should perform knn search, k=0', () {
      final k = 0;
      final searchRadius = 3;
      final result =
          searcher.query(Vector.fromList([23, 12, 34]), k, searchRadius);

      expect(result, hasLength(k));
    });
  });
}
