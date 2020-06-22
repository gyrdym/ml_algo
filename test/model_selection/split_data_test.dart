import 'package:ml_algo/src/model_selection/exception/empty_ratio_collection_exception.dart';
import 'package:ml_algo/src/model_selection/exception/invalid_ratio_sum_exception.dart';
import 'package:ml_algo/src/model_selection/exception/outranged_ratio_exception.dart';
import 'package:ml_algo/src/model_selection/exception/too_small_ratio_exception.dart';
import 'package:ml_algo/src/model_selection/split_data.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:test/test.dart';

void main() {
  group('splitData', () {
    final header = ['feature_1', 'feature_3', 'feature_3'];
    final source = [
      ['feature_1', 'feature_3', 'feature_3'],
      [     100.00,        null,      200.33],
      [      -2221,        1002,       70009],
      [       9008,       10006,        null],
      [       7888,       10002,      300918],
      [     500981,       29918,     5008.55],
    ];
    final data = DataFrame(source);

    test('should throw an exception if ratio collection is empty', () {
      expect(() => splitData(data, []),
          throwsA(isA<EmptyRatioCollectionException>()));
    });

    test('should throw an exception if at least one ratio value is negative', () {
      expect(() => splitData(data, [0.2, -0.3]),
          throwsA(isA<OutRangedRatioException>()));
    });

    test('should throw an exception if at least one ratio value is zero', () {
      expect(() => splitData(data, [0.2, 0]),
          throwsA(isA<OutRangedRatioException>()));
    });

    test('should throw an exception if at least one ratio value is equal '
        'to 1', () {
      expect(() => splitData(data, [1, 0.3]),
          throwsA(isA<OutRangedRatioException>()));
    });

    test('should throw an exception if at least one ratio value is greater '
        'than 1', () {
      expect(() => splitData(data, [100, 0.3]),
          throwsA(isA<OutRangedRatioException>()));
    });

    test('should split data', () {
      final splits = splitData(data, [0.2, 0.3])
          .toList();
      
      expect(splits, hasLength(3));
      expect(splits[0].header, header);
      expect(splits[0].rows, [
        [100.00, null, 200.33],
      ]);
      expect(splits[1].header, header);
      expect(splits[1].rows, [
        [-2221,  1002, 70009],
        [ 9008, 10006,  null],
      ]);
      expect(splits[2].header, header);
      expect(splits[2].rows, [
        [  7888, 10002,  300918],
        [500981, 29918, 5008.55],
      ]);
    });

    test('should split data, case 2', () {
      final splits = splitData(data, [0.2, 0.2, 0.2, 0.2])
          .toList();

      expect(splits, hasLength(5));
      expect(splits[0].header, header);
      expect(splits[0].rows, [
        [100.00, null, 200.33],
      ]);
      expect(splits[1].header, header);
      expect(splits[1].rows, [
        [-2221,  1002, 70009],
      ]);
      expect(splits[2].header, header);
      expect(splits[2].rows, [
        [ 9008, 10006,  null],
      ]);
      expect(splits[3].header, header);
      expect(splits[3].rows, [
        [  7888, 10002,  300918],
      ]);
      expect(splits[4].header, header);
      expect(splits[4].rows, [
        [500981, 29918, 5008.55],
      ]);
    });

    test('should throw exception if there is a too small ratio, case 1', () {
      expect(() => splitData(data, [0.2, 0.3, 0.01]),
          throwsA(isA<TooSmallRatioException>()));
    });

    test('should throw exception if there is a too small ratio, case 2', () {
      expect(() => splitData(data, [0.2, 0.3, 0.1]),
          throwsA(isA<TooSmallRatioException>()));
    });

    test('should split data into two parts, first part is less than the '
        'second one', () {
      final splits = splitData(data, [0.2])
          .toList();

      expect(splits, hasLength(2));
      expect(splits[0].header, header);
      expect(splits[0].rows, [
        [100.00, null, 200.33],
      ]);
      expect(splits[1].header, header);
      expect(splits[1].rows, [
        [ -2221,  1002,   70009],
        [  9008, 10006,    null],
        [  7888, 10002,  300918],
        [500981, 29918, 5008.55],
      ]);
    });

    test('should split data into two parts, first part is less than the '
        'second one, case 2', () {
      final splits = splitData(data, [0.25])
          .toList();

      expect(splits, hasLength(2));
      expect(splits[0].header, header);
      expect(splits[0].rows, [
        [100.00, null, 200.33],
        [-2221,  1002,  70009],
      ]);
      expect(splits[1].header, header);
      expect(splits[1].rows, [
        [  9008, 10006,    null],
        [  7888, 10002,  300918],
        [500981, 29918, 5008.55],
      ]);
    });

    test('should split data into two parts, first part is greater than the '
        'second one', () {
      final splits = splitData(data, [0.9])
          .toList();

      expect(splits, hasLength(2));
      expect(splits[0].header, header);
      expect(splits[0].rows, [
        [100.00,  null, 200.33],
        [ -2221,  1002,  70009],
        [  9008, 10006,   null],
        [  7888, 10002, 300918],
      ]);
      expect(splits[1].header, header);
      expect(splits[1].rows, [
        [500981, 29918, 5008.55],
      ]);
    });

    test('should split data into two parts, first part is greater than the '
        'second one, case 2', () {
      final splits = splitData(data, [0.95])
          .toList();

      expect(splits, hasLength(2));
      expect(splits[0].header, header);
      expect(splits[0].rows, [
        [100.00,  null, 200.33],
        [ -2221,  1002,  70009],
        [  9008, 10006,   null],
        [  7888, 10002, 300918],
      ]);
      expect(splits[1].header, header);
      expect(splits[1].rows, [
        [500981, 29918, 5008.55],
      ]);
    });

    test('should throw exception if ratios sum is equal to 1, '
        '2 elements', () {
      expect(() => splitData(data, [0.5, 0.5]),
          throwsA(isA<InvalidRatioSumException>()));
    });

    test('should throw exception if ratios sum is equal to 1, '
        '3 elements', () {
      expect(() => splitData(data, [0.3, 0.3, 0.4]),
          throwsA(isA<InvalidRatioSumException>()));
    });

    test('should throw exception if ratios sum is greater than 1, '
        '2 elements', () {
      expect(() => splitData(data, [0.5, 0.6]),
          throwsA(isA<InvalidRatioSumException>()));
    });

    test('should throw exception if ratios sum is greater than 1, '
        '3 elements', () {
      expect(() => splitData(data, [0.3, 0.5, 0.4]),
          throwsA(isA<InvalidRatioSumException>()));
    });
  });
}
