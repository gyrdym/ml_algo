import 'package:dart_ml/dart_ml.dart' show Norm, RegularVector;
import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

void main() {
  RegularVector vector1;
  RegularVector vector2;

  group('Regular vector initialization.\n', () {
    tearDown(() {
      vector1 = null;
    });

    test('from dynamic-length list:\n', () {
      vector1 = new RegularVector.from([1.0, 2.0, 3.0]);

      expect(vector1.length, equals(3));

      expect(vector1[0], equals(1.0));

      expect(vector1, equals([1.0, 2.0, 3.0]));
      expect(() => vector1[3], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
    });

    test('from fixed-length list:\n', () {
      List<double> source = new List<double>.filled(3, 1.0);
      vector1 = new RegularVector.from(source);

      expect(vector1.length, equals(3));
      expect(vector1, equals([1.0, 1.0, 1.0]));
      expect(() => vector1[3], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
    });

    test('Assign value to vector element:\n', () {
      List<double> source = new List<double>.filled(3, 1.0);
      vector1 = new RegularVector.from(source);
      vector1[1] = 45.0;

      expect(vector1.length, equals(3));
      expect(vector1, equals([1.0, 45.0, 1.0]));
      expect(() => vector1[3], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
    });

    test('Vector dimension (length) manipulation:\n', () {
      List<double> source = new List<double>.filled(5, 1.0);
      vector1 = new RegularVector.from(source);

      expect(vector1.length, equals(5));
      expect(vector1.length, equals(vector1.length));

      vector1.length = 10;

      expect(vector1[8], isZero);
      expect(vector1, equals([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    });
  });

  group('Regular vector operations (not inplace).\n', () {
    setUp(() {
      vector1 = new RegularVector.from([1.0, 2.0, 3.0, 4.0]);
      vector2 = new RegularVector.from([1.0, 2.0, 3.0, 4.0]);
    });

    tearDown(() {
      vector1 = null;
      vector2 = null;
    });

    test('vectors addition:\n', () {
      RegularVector result = vector1 + vector2;

      expect(result != vector1, isTrue);
      expect(result.length, equals(4));
      expect(result, equals([2.0, 4.0, 6.0, 8.0]));
      expect(() => result[4], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('vectors subtraction:\n', () {
      RegularVector result = vector1 - vector2;

      expect(result != vector1, isTrue);
      expect(result.length, equals(4));
      expect(result, equals([0.0, 0.0, 0.0, 0.0]));
      expect(() => result[4], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('vectors multiplication (algebraic format):\n', () {
      RegularVector result = vector1 * vector2;

      expect(result != vector1, isTrue);
      expect(result.length, equals(4));
      expect(result, equals([1.0, 4.0, 9.0, 16.0]));
      expect(() => result[4], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('vectors division (It\'s illegal operation for vectors from the math side, but it can be useful in the future):\n', () {
      RegularVector result = vector1 / vector2;

      expect(result != vector1, isTrue);
      expect(result.length, equals(4));
      expect(result, equals([1.0, 1.0, 1.0, 1.0]));
      expect(() => result[4], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('element-wise vector power:\n', () {
      RegularVector result = vector1.intPow(3);

      expect(result != vector1, isTrue);
      expect(result.length, equals(4));
      expect(result, equals([1.0, 8.0, 27.0, 64.0]));
      expect(() => result[4], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('element-wise vector power, inplace:\n', () {
      RegularVector result = vector1.intPow(3, inPlace: true);

      expect(result, same(vector1));
      expect(vector1.length, equals(4));
      expect(vector1, equals([1.0, 8.0, 27.0, 64.0]));
      expect(() => vector1[4], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
    });

    test('vector multiplication (scalar format):\n', () {
      double result = vector1.vectorScalarMult(vector2);

      expect(result, equals(30.0));
    });

    test('vector and scalar multiplication:\n', () {
      RegularVector result = vector1.scalarMult(2.0);

      expect(result != vector1, isTrue);
      expect(result.length, equals(4));
      expect(result, equals([2.0, 4.0, 6.0, 8.0]));
      expect(() => result[4], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('vector and scalar multiplication, inplace:\n', () {
      RegularVector result = vector1.scalarMult(2.0, inPlace: true);

      expect(result, same(vector1));
      expect(vector1.length, equals(4));
      expect(vector1, equals([2.0, 4.0, 6.0, 8.0]));
      expect(() => vector1[4], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
    });

    test('vector and scalar division:\n', () {
      RegularVector result = vector1.scalarDivision(2.0);

      expect(result != vector1, isTrue);
      expect(result.length, equals(4));
      expect(result, equals([0.5, 1.0, 1.5, 2.0]));
      expect(() => result[4], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('vector and scalar division, inplace:\n', () {
      RegularVector result = vector1.scalarDivision(2.0, inPlace: true);

      expect(result, same(vector1));
      expect(vector1.length, equals(4));
      expect(vector1, equals([0.5, 1.0, 1.5, 2.0]));
      expect(() => vector1[4], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
    });

    test('add a scalar to a vector:\n', () {
      RegularVector result = vector1.scalarAddition(13.0);

      expect(result != vector1, isTrue);
      expect(result.length, equals(4));
      expect(result, equals([14.0, 15.0, 16.0, 17.0]));
      expect(() => result[4], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('add a scalar to a vector, inplace:\n', () {
      RegularVector result = vector1.scalarAddition(13.0, inPlace: true);

      expect(result, same(vector1));
      expect(vector1.length, equals(4));
      expect(vector1, equals([14.0, 15.0, 16.0, 17.0]));
      expect(() => vector1[4], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
    });

    test('subtract a scalar from a vector:\n', () {
      RegularVector result = vector1.scalarSubtraction(13.0);

      expect(result != vector1, isTrue);
      expect(result.length, equals(4));
      expect(result, equals([-12.0, -11.0, -10.0, -9.0]));
      expect(() => result[4], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('subtract a scalar from a vector, inplace:\n', () {
      RegularVector result = vector1.scalarSubtraction(13.0, inPlace: true);

      expect(result, same(vector1));
      expect(vector1.length, equals(4));
      expect(vector1, equals([-12.0, -11.0, -10.0, -9.0]));
      expect(() => vector1[4], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
    });

    test('find the euclidean distance between two identical vectors:\n', () {
      double distance = vector1.distanceTo(vector2);

      expect(distance, equals(0.0), reason: 'Wrong vector distance calculation');
    });

    test('find the euclidean distance between two different vectors:\n', () {
      vector1 = new RegularVector.from([23.0, 34.0, 12.0, 10.0]);
      vector2 = new RegularVector.from([100.0, 200.0, 300.0, 55.0]);

      expect(vector1.distanceTo(vector2), equals(344.1714688930505), reason: 'Wrong vector distance calculation');
    });

    test('find the euclidean norm of a vector', () {
      expect(vector1.norm(), equals(5.477225575051661), reason: 'Wrong norm calculation');
    });

    test('find the manhattan norm of a vector', () {
      expect(vector1.norm(Norm.MANHATTAN), equals(10.0), reason: 'Wrong norm calculation');
    });
  });
}