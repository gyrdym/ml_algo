import 'package:ml_algo/src/model_selection/split_indices_provider/k_fold_data_splitter.dart';
import 'package:test/test.dart';

void main() {
  group('KFoldDataSplitter', () {
    void testKFoldSplitter(int numOfFold, int numOfObservations,
        Iterable<Iterable<int>> expected) {
      test('should return proper groups of indices if number of folds is '
          '$numOfFold and number of observations is $numOfObservations', () {
        final splitter = KFoldDataSplitter(numOfFold);
        expect(splitter.split(numOfObservations), equals(expected));
      });
    }

    test('should throw an exception if passed number of folds is equal to '
        '0', () {
      expect(() => KFoldDataSplitter(0), throwsRangeError);
    });

    test('should throw an exception if passed number of folds is equal to '
        '1', () {
      expect(() => KFoldDataSplitter(1), throwsRangeError);
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
      [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11],
    ]);

    testKFoldSplitter(5, 37, [
      [0, 1, 2, 3, 4, 5, 6],
      [7, 8, 9, 10, 11, 12, 13],
      [14, 15, 16, 17, 18, 19, 20],
      [21, 22, 23, 24, 25, 26, 27, 28],
      [29, 30, 31, 32, 33, 34, 35, 36],
    ]);

    test('should throws a range error if number of observations is 0', () {
      final splitter = KFoldDataSplitter(3);
      expect(() => splitter.split(0), throwsRangeError);
    });

    test('should throws a range error if number of observations is less than'
        'number of folds', () {
      final splitter = KFoldDataSplitter(9);
      expect(() => splitter.split(8), throwsRangeError);
    });
  });
}
