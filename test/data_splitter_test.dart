import 'package:dart_ml/src/data_splitters/k_fold_splitter.dart';
import 'package:dart_ml/src/data_splitters/leave_p_out_splitter.dart';

import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

void main() {
  group('Splitters test.\n', () {
    test('K fold splitter test', () {
      KFoldSplitter splitter;

      splitter = new KFoldSplitter();
      expect(splitter.split(12), equals([[0,3],[3,6],[6,8],[8,10],[10,12]]));

      splitter = new KFoldSplitter(numberOfFolds: 4);
      expect(splitter.split(12), equals([[0,3],[3,6],[6,9],[9,12]]));

      splitter = new KFoldSplitter(numberOfFolds: 3);
      expect(splitter.split(12), equals([[0,4],[4,8],[8,12]]));

      splitter = new KFoldSplitter(numberOfFolds: 1);
      expect(splitter.split(12), equals([[0,12]]));

      splitter = new KFoldSplitter(numberOfFolds: 12);
      expect(splitter.split(12), equals([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12]]));

      splitter = new KFoldSplitter(numberOfFolds: 3);
      expect(splitter.split(67), equals([[0,23],[23,45],[45,67]]));

      splitter = new KFoldSplitter(numberOfFolds: 3);
      expect(() => splitter.split(0), throwsRangeError);

      splitter = new KFoldSplitter(numberOfFolds: 9);
      expect(() => splitter.split(8), throwsRangeError);
    });

    test('Leave P out splitter test', () {
      LeavePOutSplitter splitter;

      splitter = new LeavePOutSplitter(p: 2);
      expect(splitter.split(4), equals([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]));
      expect(splitter.split(5), equals([[0,1], [0,2], [0,3], [0,4], [1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]));

      splitter = new LeavePOutSplitter();
      expect(splitter.split(4), equals([[0], [1], [2], [3], [4]]));
    });
  });
}