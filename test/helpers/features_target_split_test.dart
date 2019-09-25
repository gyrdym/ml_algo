import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:test/test.dart';

void main() {
  group('featuresTargetSplit', () {
    test('should split given dataset by target index into two parts - features '
        'dataframe and target dataframe', () {
      final dataset = DataFrame(<Iterable<num>>[
        [ 0,  1,  2,  4,  5],
        [10, 20, 30, 40, 50],
        [66, 77, 88, 99, 11],
        [11, 22, 33, 44, 55],
      ], headerExists: false);

      final splits = featuresTargetSplit(dataset, targetIndices: [3]).toList();

      expect(splits[0].toMatrix(), equals([
        [ 0,  1,  2,  5],
        [10, 20, 30, 50],
        [66, 77, 88, 11],
        [11, 22, 33, 55],
      ]));

      expect(splits[1].toMatrix(), equals([
        [ 4],
        [40],
        [99],
        [44],
      ]));
    });

    test('should split given dataset by target indices into two parts - '
        'features dataframe and target dataframe', () {
      final dataset = DataFrame(<Iterable<num>>[
        [ 0,  1,  2,  4,  5],
        [10, 20, 30, 40, 50],
        [66, 77, 88, 99, 11],
        [11, 22, 33, 44, 55],
      ], headerExists: false);

      final splits = featuresTargetSplit(dataset, targetIndices: [0, 3])
          .toList();

      expect(splits[0].toMatrix(), equals([
        [ 1,  2,  5],
        [20, 30, 50],
        [77, 88, 11],
        [22, 33, 55],
      ]));

      expect(splits[1].toMatrix(), equals([
        [ 0,  4],
        [10, 40],
        [66, 99],
        [11, 44],
      ]));
    });

    test('should split given dataset just by unique target indices', () {
      final dataset = DataFrame(<Iterable<num>>[
        [ 0,  1,  2,  4,  5],
        [10, 20, 30, 40, 50],
        [66, 77, 88, 99, 11],
        [11, 22, 33, 44, 55],
      ], headerExists: false);

      final splits = featuresTargetSplit(
          dataset, targetIndices: [0, 3, 0, 0, 3]).toList();

      expect(splits[0].toMatrix(), equals([
        [ 1,  2,  5],
        [20, 30, 50],
        [77, 88, 11],
        [22, 33, 55],
      ]));

      expect(splits[1].toMatrix(), equals([
        [ 0,  4],
        [10, 40],
        [66, 99],
        [11, 44],
      ]));
    });

    test('should throw an error if there is an outranged index while accessing '
        'the target split', () {
      final dataset = DataFrame(<Iterable<num>>[
        [ 0,  1,  2,  4,  5],
        [10, 20, 30, 40, 50],
        [66, 77, 88, 99, 11],
        [11, 22, 33, 44, 55],
      ], headerExists: false);

      final splits = featuresTargetSplit(
          dataset, targetIndices: [0, 30]).toList();

      expect(splits, hasLength(2));

      expect(splits[0].toMatrix(), equals([
        [ 1,  2,  4,  5],
        [20, 30, 40, 50],
        [77, 88, 99, 11],
        [22, 33, 44, 55],
      ]));

      expect(() => splits[1].toMatrix(), throwsRangeError);
    });
  });
}
