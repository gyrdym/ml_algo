import 'package:dart_ml/src/data_splitters/k_fold_splitter.dart';

import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

void main() {
  group('Splitters test.\n', () {
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