import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_factory/observations_splitter/samples_splitter_impl.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('SamplesSplitterImpl', () {
    test('should split given matrix into two parts: first part should contain '
        'values less than the splitting value, right part should contain '
        'values greater than the splitting value', () {
      final samples = Matrix.fromList([
        [111, 2,  30, 4],
        [1,   32, 10, 44],
        [11,  22, 10, 14],
        [33,  12, 5,  55],
        [0,   20, 60, 10],
      ]);
      final splittingColumnIdx = 2;
      final splittingValue = 10.0;
      final splitter = const SamplesSplitterImpl();
      final actual = splitter.split(samples, splittingColumnIdx,
          splittingValue);
      expect(actual, equals([
        [
          [33,  12, 5,  55],
        ],
        [
          [111, 2,  30, 4],
          [1,   32, 10, 44],
          [11,  22, 10, 14],
          [0,   20, 60, 10],
        ],
      ]));
    });

    test('should split given matrix into two parts if all the values are '
        'greater or equal than the splitting value: the first part should be '
        'empty', () {
      final samples = Matrix.fromList([
        [111, 2,  30, 4],
        [1,   32, 10, 44],
        [11,  22, 10, 14],
        [33,  12, 500,  55],
        [0,   20, 60, 10],
      ]);
      final splittingColumnIdx = 2;
      final splittingValue = 10.0;
      final splitter = const SamplesSplitterImpl();
      final actual = splitter.split(samples, splittingColumnIdx,
          splittingValue);
      expect(actual, equals([
        <double>[],
        [
          [111, 2,  30, 4],
          [1,   32, 10, 44],
          [11,  22, 10, 14],
          [33,  12, 500,  55],
          [0,   20, 60, 10],
        ],
      ]));
    });

    test('should split given matrix into two parts if splitting value is '
        '0', () {
      final samples = Matrix.fromList([
        [111, 2,  30, 4],
        [1,   32, -10, 44],
        [11,  22, 0, 14],
        [33,  12, 500,  55],
        [0,   20, -60, 10],
      ]);
      final splittingColumnIdx = 2;
      final splittingValue = 0.0;
      final splitter = const SamplesSplitterImpl();
      final actual = splitter.split(samples, splittingColumnIdx,
          splittingValue);
      expect(actual, equals([
        [
          [1,   32, -10, 44],
          [0,   20, -60, 10],
        ],
        [
          [111, 2,  30, 4],
          [11,  22, 0, 14],
          [33,  12, 500,  55],
        ],
      ]));
    });

    test('should split given matrix into two parts if all the values are '
        'less than the splitting value: the second part should be empty', () {
      final samples = Matrix.fromList([
        [111, 2,  30, 4],
        [1,   32, 10, 44],
        [11,  22, 10, 14],
        [33,  12, 500,  55],
        [0,   20, 60, 10],
      ]);
      final splittingColumnIdx = 2;
      final splittingValue = 1000.0;
      final splitter = const SamplesSplitterImpl();
      final actual = splitter.split(samples, splittingColumnIdx,
          splittingValue);
      expect(actual, equals([
        [
          [111, 2,  30, 4],
          [1,   32, 10, 44],
          [11,  22, 10, 14],
          [33,  12, 500,  55],
          [0,   20, 60, 10],
        ],
        <double>[],
      ]));
    });

    test('should split given matrix into two parts if splitting column index is'
        ' 0', () {
      final samples = Matrix.fromList([
        [111, 2,  30, 4],
        [1,   32, 10, 44],
        [11,  22, 10, 14],
        [33,  12, 500,  55],
        [0,   20, 60, 10],
      ]);
      final splittingColumnIdx = 0;
      final splittingValue = 2.0;
      final splitter = const SamplesSplitterImpl();
      final actual = splitter.split(samples, splittingColumnIdx,
          splittingValue);
      expect(actual, equals([
        [
          [1,   32, 10, 44],
          [0,   20, 60, 10],
        ],
        [
          [111, 2,  30, 4],
          [11,  22, 10, 14],
          [33,  12, 500,  55],
        ],
      ]));
    });

    test('should split given matrix into two parts if splitting column is the '
        'last column', () {
      final samples = Matrix.fromList([
        [111, 2,  30, 4],
        [1,   32, 10, 44],
        [11,  22, 10, 14],
        [33,  12, 500,  55],
        [0,   20, 60, 10],
      ]);
      final splittingColumnIdx = 3;
      final splittingValue = 20.0;
      final splitter = const SamplesSplitterImpl();
      final actual = splitter.split(samples, splittingColumnIdx,
          splittingValue);
      expect(actual, equals([
        [
          [111, 2,  30, 4],
          [11,  22, 10, 14],
          [0,   20, 60, 10],
        ],
        [
          [1,   32, 10, 44],
          [33,  12, 500,  55],
        ],
      ]));
    });
  });
}
