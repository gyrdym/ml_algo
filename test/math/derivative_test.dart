import 'package:dart_ml/src/interface.dart';
import 'package:dart_ml/src/implementation.dart';
import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

void main() {
  DerivativeFinder finder;

  group('Derivative finder', () {
    setUp(() {
      finder = MathUtils.createDerivativeFinder();
//      finder.configure(numberOfArguments, 0.00001, function, metric);
    });

    test('should return an approximately value of the derivative', () {

//      expect(start <= value && value < end, isTrue);
    });
  });
}