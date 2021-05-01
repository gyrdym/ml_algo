import 'dart:math';

import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_impl.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

class RandomMock extends Mock implements Random {}

void main() {
  const maxEpoch = 400;

  late Randomizer randomizer;

  group('Randomizer (without predefined generator)', () {
    setUp(() {
      randomizer = RandomizerImpl();
    });

    test(
        'should return interval [lowerBound, upperBound] if `lowerBound` and `upperBound` differ from each other by 1',
        () {
      final lowerBound = 0;
      final upperBound = 1;
      final interval = randomizer.getIntegerInterval(lowerBound, upperBound);

      expect(interval, equals([0, 1]));
    });

    test(
        'should return interval [lowerBound, upperBound] if `lowerBound` and `upperBound` differ from each other by '
        'exactly the `intervalLength` value, zero `lowerBound` case', () {
      final lowerBound = 0;
      final upperBound = 1;
      final interval = randomizer.getIntegerInterval(lowerBound, upperBound,
          intervalLength: 1);

      expect(interval, equals([0, 1]));
    });

    test(
        'should return interval [lowerBound, upperBound] if `lowerBound` and `upperBound` differ from each other by '
        'exactly the `intervalLength` value, zero `lowerBound` case, double check',
        () {
      final lowerBound = 0;
      final upperBound = 2;
      final interval = randomizer.getIntegerInterval(lowerBound, upperBound,
          intervalLength: 2);

      expect(interval, equals([0, 2]));
    });

    test(
        'should return interval [lowerBound, upperBound] if `lowerBound` and `upperBound` differ from each other by 1, '
        'non-zero `lowerBound` case', () {
      final lowerBound = 1;
      final upperBound = 2;
      final interval = randomizer.getIntegerInterval(lowerBound, upperBound);

      expect(interval, equals([1, 2]));
    });

    test(
        'should return interval [lowerBound, upperBound] if `lowerBound` and `upperBound` differ from each other by '
        'exactly the `intervalLength` value, non-zero `lowerBound` case', () {
      final lowerBound = 3;
      final upperBound = 5;
      final interval = randomizer.getIntegerInterval(lowerBound, upperBound,
          intervalLength: 2);

      expect(interval, equals([3, 5]));
    });

    test('should throw an error if `intervalLength` exceeds the total interval',
        () {
      final lowerBound = 2;
      final upperBound = 4;

      expect(
          () => randomizer.getIntegerInterval(lowerBound, upperBound,
              intervalLength: 3),
          throwsRangeError);
    });

    test('should return a proper integer value from the given interval', () {
      final start = 2;
      final end = 13;

      for (var i = 0; i < maxEpoch; i++) {
        final value = randomizer.getIntegerFromInterval(start, end);
        expect(start <= value && value < end, isTrue);
      }
    });

    test('should return a proper integer interval constrained by given bounds',
        () {
      final intervalLength = 3;
      final start = 6;
      final end = 17;

      for (var i = 0; i < maxEpoch; i++) {
        final interval = randomizer.getIntegerInterval(start, end,
            intervalLength: intervalLength);

        expect(start <= interval.first && interval.first < end, isTrue);
        expect(start < interval.last && interval.last < end, isTrue);
        expect(interval.first < interval.last, isTrue);
      }
    });

    test(
        'should return range error if start and end values are equal (integer interval generation)',
        () {
      final start = 0;
      final end = 0;

      expect(() => randomizer.getIntegerInterval(start, end), throwsRangeError);
    });

    test('should always return a zero in case of [0, 1] passed interval', () {
      final start = 0;
      final end = 1;

      for (var i = 0; i < maxEpoch; i++) {
        final value = randomizer.getIntegerFromInterval(start, end);
        expect(value, isZero);
      }
    });

    test(
        'should return range error if start and end values are equal (integer value generation)',
        () {
      final start = 1;
      final end = 1;

      expect(() => randomizer.getIntegerFromInterval(start, end),
          throwsRangeError);
    });

    test('should return a proper double value from the given interval', () {
      final start = 5.3;
      final end = 123.4;

      for (var i = 0; i < maxEpoch; i++) {
        final value = randomizer.getDoubleFromInterval(start, end);
        expect(start <= value && value < end, isTrue);
      }
    });

    test(
        'should return range error if start and end values are equal (double value generation)',
        () {
      final start = 1.0;
      final end = 1.0;

      expect(
          () => randomizer.getDoubleFromInterval(start, end), throwsRangeError);
    });
  });
}
