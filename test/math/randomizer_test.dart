import 'package:dart_ml/src/interface.dart';
import 'package:dart_ml/src/implementation.dart';
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
      int start = 2;
      int end = 13;

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

        List<int> interval = randomizer.getIntegerInterval(start, end);
        expect(start <= interval.first && interval.first < end, isTrue);
        expect(start < interval.last && interval.last < end, isTrue);
        expect(interval.first < interval.last, isTrue);
      }

      start = 6;
      end = 8;

      expect(randomizer.getIntegerInterval(start, end), [6, 7]);
    });

    test('should throw a range error if difference of lower and upper bounds is 1', () {
      int start = 6;
      int end = 7;

      expect(() => randomizer.getIntegerInterval(start, end), throwsRangeError);
    });

    test('should return range error if start and end values are equal (integer interval generation)', () {
      int start = 0;
      int end = 0;

      expect(() => randomizer.getIntegerInterval(start, end), throwsRangeError);
    });

    test('should always return a zero in case of [0, 1] passed interval', () {
      int start = 0;
      int end = 1;

      for (int i = 0; i < MAX_EPOCH; i++) {
        int value = randomizer.getIntegerFromInterval(start, end);
        expect(value, isZero);
      }
    });

    test('should return range error if start and end values are equal (integer value generation)', () {
      int start = 1;
      int end = 1;

      expect(() => randomizer.getIntegerFromInterval(start, end), throwsRangeError);
    });

    test('should return a proper double value from the given interval', () {
      double start = 5.3;
      double end = 123.4;

      for (int i = 0; i < MAX_EPOCH; i++) {
        double value = randomizer.getDoubleFromInterval(start, end);
        expect(start <= value && value < end, isTrue);
      }
    });

    test('should return range error if start and end values are equal (double value generation)', () {
      double start = 1.0;
      double end = 1.0;

      expect(() => randomizer.getDoubleFromInterval(start, end), throwsRangeError);
    });
  });
}