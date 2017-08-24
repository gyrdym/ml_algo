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