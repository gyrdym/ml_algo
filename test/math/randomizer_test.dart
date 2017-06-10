import 'package:dart_ml/src/math/misc/randomizer/implementation/randomizer.dart';
import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

void main() {
  const int MAX_EPOCH = 200;

  RandomizerImpl randomizer;

  group('Randomizer ', () {
    setUp(() {
      randomizer = new RandomizerImpl();
    });

    test('should return a proper integer value from the given interval', () {
      int start = 2;
      int end = 13;

      for (int i = 0; i < MAX_EPOCH; i++) {
        int value = randomizer.getIntegerFromInterval(start, end);
        expect(start <= value && value < end, isTrue);
      }
    });

    test('should return a proper double value from the given interval', () {
      double start = 5.3;
      double end = 123.4;

      for (int i = 0; i < MAX_EPOCH; i++) {
        double value = randomizer.getDoubleFromInterval(start, end);
        expect(start <= value && value < end, isTrue);
      }
    });
  });
}