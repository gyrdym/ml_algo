import 'package:dart_ml/src/data_splitters/simple_splitter.dart';
import 'package:dart_ml/src/data_splitters/k_fold_splitter.dart';
import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

void main() {
  group('Splitters test.\n', () {
    test('Simple splitter test: ', () {
      SimpleDataSplitter simpleSplitter = new SimpleDataSplitter(ratio: .7);

      List<List<int>> ranges = simpleSplitter.split(10);
      List<int> trainRange = ranges[0];
      List<int> testRange = ranges[1];

      expect(trainRange.toSet(), equals([0, 1, 2, 3, 4, 5, 6].toSet()));
      expect(testRange.toSet(), equals([7, 8, 9].toSet()));

      ranges = simpleSplitter.split(8);
      trainRange = ranges[0];
      testRange = ranges[1];

      expect(trainRange.toSet(), equals([0, 1, 2, 3, 4, 5].toSet()));
      expect(testRange.toSet(), equals([6, 7].toSet()));
    });

    test('K fold splitter test', () {
      KFoldSplitter splitter = new KFoldSplitter();

      expect(splitter.split(12), equals([[0,3],[3,6],[6,8],[8,10],[10,12]]));
      expect(splitter.split(12, numberOfFolds: 4), equals([[0,3],[3,6],[6,9],[9,12]]));
      expect(splitter.split(12, numberOfFolds: 3), equals([[0,4],[4,8],[8,12]]));
      expect(splitter.split(12, numberOfFolds: 1), equals([[0,12]]));
      expect(splitter.split(12, numberOfFolds: 12), equals([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12]]));
      expect(splitter.split(67, numberOfFolds: 3), equals([[0,23],[23,45],[45,67]]));
      expect(() => splitter.split(0, numberOfFolds: 3), throwsRangeError);
      expect(() => splitter.split(8, numberOfFolds: 9), throwsRangeError);
    });
  });
}