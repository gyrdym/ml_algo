import 'package:dart_ml/src/core/interface.dart';
import 'package:dart_ml/src/core/implementation.dart';
import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

void main() {
  const int MAX_EPOCH = 200;

  Randomizer randomizer;

  group('Randomizer ', () {
    setUp(() {
      randomizer = MathUtils.createRandomizer();
    });

    test('should return a proper integer value from the given interval', () {
      final start = 2;
      final end = 13;

      for (int i = 0; i < MAX_EPOCH; i++) {
        int value = randomizer.getIntegerFromInterval(start, end);
        expect(start <= value && value < end, isTrue);
      }
    });

    test('should return a proper integer interval constrained by given bounds', () {
      int start;
      int end;

      for (int i = 0; i < MAX_EPOCH; i++) {
        start = 6;
        end = 17;

        final interval = randomizer.getIntegerInterval(start, end);
        expect(start <= interval.first && interval.first < end, isTrue);
        expect(start < interval.last && interval.last < end, isTrue);
        expect(interval.first < interval.last, isTrue);
      }

      start = 6;
      end = 8;

      expect(randomizer.getIntegerInterval(start, end), [6, 7]);
    });

    test('should throw a range error if difference of lower and upper bounds is 1', () {
      final start = 6;
      final end = 7;

      expect(() => randomizer.getIntegerInterval(start, end), throwsRangeError);
    });

    test('should return range error if start and end values are equal (integer interval generation)', () {
      final start = 0;
      final end = 0;

      expect(() => randomizer.getIntegerInterval(start, end), throwsRangeError);
    });

    test('should always return a zero in case of [0, 1] passed interval', () {
      final start = 0;
      final end = 1;

      for (int i = 0; i < MAX_EPOCH; i++) {
        final value = randomizer.getIntegerFromInterval(start, end);
        expect(value, isZero);
      }
    });

    test('should return range error if start and end values are equal (integer value generation)', () {
      final start = 1;
      final end = 1;

      expect(() => randomizer.getIntegerFromInterval(start, end), throwsRangeError);
    });

    test('should return a proper double value from the given interval', () {
      final start = 5.3;
      final end = 123.4;

      for (int i = 0; i < MAX_EPOCH; i++) {
        final value = randomizer.getDoubleFromInterval(start, end);
        expect(start <= value && value < end, isTrue);
      }
    });

    test('should return range error if start and end values are equal (double value generation)', () {
      final start = 1.0;
      final end = 1.0;

      expect(() => randomizer.getDoubleFromInterval(start, end), throwsRangeError);
    });
  });
}