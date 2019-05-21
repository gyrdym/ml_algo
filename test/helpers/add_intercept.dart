import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('addInterceptIf', () {
    test('should return the same matrix if `fitIntercept` is `false`', () {
      final observations = Matrix.fromList([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
      ]);
      final actual = addInterceptIf(false, observations, 1.0);
      expect(actual, same(observations));
    });

    test('should return new matrix with extra column inserted in the beginning '
        'which has each element equals `2` if `fitIntercept` is `true` and '
        '`interceptSacle` is `2`', () {
      final observations = Matrix.fromList([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
      ]);
      final actual = addInterceptIf(true, observations, 2.0);
      expect(actual, equals([
        [2.0, 1.0, 2.0, 3.0, 4.0],
        [2.0, 5.0, 6.0, 7.0, 8.0],
      ]));
    });

    test('should return new matrix with extra column inserted in the beginning '
        'which has each element equals `0` if `fitIntercept` is `true` and '
        '`interceptSacle` is `0`', () {
      final observations = Matrix.fromList([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
      ]);
      final actual = addInterceptIf(true, observations, 0.0);
      expect(actual, equals([
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 5.0, 6.0, 7.0, 8.0],
      ]));
    });
  });
}
