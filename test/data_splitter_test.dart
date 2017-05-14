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

      splitter = new LeavePOutSplitter();
      expect(splitter.split(4), equals([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]));
      expect(splitter.split(5), equals([[0,1], [0,2], [0,3], [0,4], [1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]));
      expect(splitter.split(13), equals([
         [0,1], [0,2], [0,3], [0,4], [0,5], [0,6], [0,7], [0,8], [0,9], [0,10], [0,11], [0,12]
        ,[1,2], [1,3], [1,4], [1,5], [1,6], [1,7], [1,8], [1,9], [1,10], [1,11], [1,12]
        ,[2,3], [2,4], [2,5], [2,6], [2,7], [2,8], [2,9], [2,10], [2,11], [2,12]
        ,[3,4], [3,5], [3,6], [3,7], [3,8], [3,9], [3,10], [3,11], [3,12]
        ,[4,5], [4,6], [4,7], [4,8], [4,9], [4,10], [4,11], [4,12]
        ,[5,6], [5,7], [5,8], [5,9], [5,10], [5,11], [5,12]
        ,[6,7], [6,8], [6,9], [6,10], [6,11], [6,12]
        ,[7,8], [7,9], [7,10], [7,11], [7,12]
        ,[8,9], [8,10], [8,11], [8,12]
        ,[9,10], [9,11], [9,12]
        ,[10,11], [10,12]
        ,[11,12]
      ]));

      splitter = new LeavePOutSplitter(p: 3);
      expect(splitter.split(4), equals([[0,1,2], [0,1,3], [0,2,3], [1,2,3]]));

      expect(() => splitter = new LeavePOutSplitter(p: 0), throwsUnsupportedError);
      expect(() => splitter = new LeavePOutSplitter(p: 1), throwsUnsupportedError);
    });
  });
}