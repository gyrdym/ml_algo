import 'package:ml_algo/src/model_selection/split_indices_provider/k_fold_data_splitter.dart';
import 'package:test/test.dart';

void main() {
  group('KFoldIndicesProvider', () {
    void testKFoldSplitter(int numOfFold, int numOfObservations,
        Iterable<Iterable<int>> expected) {
      test(
          'should return proper groups of indices if number of folds is '
          '$numOfFold and number of observations is $numOfObservations', () {
        final splitter = KFoldIndicesProvider(numOfFold);
        expect(splitter.getIndices(numOfObservations), equals(expected));
      });
    }

    test(
        'should throw an exception if passed number of folds is equal to '
        '0', () {
      expect(() => KFoldIndicesProvider(0), throwsRangeError);
    });

    test(
        'should throw an exception if passed number of folds is equal to '
        '1', () {
      expect(() => KFoldIndicesProvider(1), throwsRangeError);
    });

    testKFoldSplitter(5, 12, [
      [0, 1],
      [2, 3],
      [4, 5],
      [6, 7, 8],
      [9, 10, 11],
    ]);

    testKFoldSplitter(4, 12, [
      [0, 1, 2],
      [3, 4, 5],
      [6, 7, 8],
      [9, 10, 11],
    ]);

    testKFoldSplitter(3, 12, [
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [8, 9, 10, 11],
    ]);

    testKFoldSplitter(12, 12, [
      [0],
      [1],
      [2],
      [3],
      [4],
      [5],
      [6],
      [7],
      [8],
      [9],
      [10],
      [11],
    ]);

    testKFoldSplitter(5, 37, [
      [0, 1, 2, 3, 4, 5, 6],
      [7, 8, 9, 10, 11, 12, 13],
      [14, 15, 16, 17, 18, 19, 20],
      [21, 22, 23, 24, 25, 26, 27, 28],
      [29, 30, 31, 32, 33, 34, 35, 36],
    ]);

    test('should throws a range error if number of observations is 0', () {
      final splitter = KFoldIndicesProvider(3);
      expect(() => splitter.getIndices(0), throwsRangeError);
    });

    test(
        'should throws a range error if number of observations is less than'
        'number of folds', () {
      final splitter = KFoldIndicesProvider(9);
      expect(() => splitter.getIndices(8), throwsRangeError);
    });
  });
}
