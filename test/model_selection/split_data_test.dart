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

    test('should throw an exception if ratio list is empty', () {
      expect(() => splitData(data, []), throwsException);
    });

    test('should throw an exception if at least one ratio value is negative', () {
      expect(() => splitData(data, [0.2, -0.3]), throwsException);
    });

    test('should throw an exception if at least one ratio value is zero', () {
      expect(() => splitData(data, [0.2, 0]), throwsException);
    });

    test('should throw an exception if at least one ratio value is equal '
        'to 1', () {
      expect(() => splitData(data, [1, 0.3]), throwsException);
    });

    test('should throw an exception if at least one ratio value is greater '
        'than 1', () {
      expect(() => splitData(data, [100, 0.3]), throwsException);
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
  });
}
