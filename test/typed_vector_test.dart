import 'dart:typed_data';
import 'package:dart_ml/dart_ml.dart' show TypedVector;
import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

void main() {
  TypedVector vector1;
  TypedVector vector2;
  TypedVector vector3;
  TypedVector vector4;

  group('Vector fundamental', () {
    tearDown(() {
      vector1 = null;
      vector2 = null;
    });

    test('Vector initialization via `from` constructor: ', () {
      //dynamic-length list
      vector1 = new TypedVector.from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

      expect(vector1[0], equals(1.0));
      expect(vector1[4], equals(5.0));
      expect(vector1[5], equals(6.0));
      expect(() => vector1[7], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);

      expect(vector1.length, equals(6));

      //fixed-length list
      vector2 = new TypedVector.from(new List.filled(11, 1.0));

      expect(vector2.length, 11);
      expect(vector2[10], 1.0);
    });

    test('Vector initialization via `fromTypedList` constructor: ', () {
      Float32x4List typedList = new Float32x4List.fromList([
        new Float32x4(1.0, 2.0, 3.0, 4.0),
        new Float32x4(5.0, 6.0, 7.0, 8.0),
        new Float32x4(9.0, 10.0, 0.0, 0.0)
      ]);

      vector1 = new TypedVector.fromTypedList(typedList);

      expect(vector1, equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0]));
      expect(() => vector1[12], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
      expect(vector1.length, equals(12));

      vector1 = new TypedVector.fromTypedList(typedList, 10);

      expect(vector1, equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]));
      expect(() => vector1[10], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
      expect(vector1.length, equals(10));
    });

    test('Vector initialization via `fill` constructor: ', () {
      vector1 = new TypedVector.filled(10, 2.0);

      expect(vector1, equals([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]));
      expect(() => vector1[11], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);

      expect(vector1.length, equals(10));
    });

    test('Vector elements updating: ', () {
      vector1 = new TypedVector.from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

      vector1[0] = 34.0;
      vector1[4] = 277.0;
      vector1[5] = 1000.0;

      expect(vector1, equals([34.0, 2.0, 3.0, 4.0, 277.0, 1000.0]));
      expect(vector1.length, equals(6));
    });

    test('Vector length updating: ', () {
      vector1 = new TypedVector.from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

      vector1.length = 8;
      expect(vector1.length, 8);

      vector1.length = 13;
      expect(vector1.length, 13);
      expect(vector1[12], equals(0.0));

      vector1.length = 0;
      expect(vector1.length, 0);
      expect(() => vector1[1], throwsRangeError);

      expect(() => vector1.length = -2, throwsRangeError);
    });
  });

  group('Vector operations.', () {
    setUp(() {
      vector1 = new TypedVector.from([1.0, 2.0, 3.0, 4.0, 5.0]);
      vector2 = new TypedVector.from([1.0, 2.0, 3.0, 4.0, 5.0]);
    });

    tearDown(() {
      vector1 = null;
      vector2 = null;
      vector3 = null;
      vector4 = null;
    });

    test('Vectors addition: ', () {
      TypedVector result = vector1 + vector2;

      expect(result, equals([2.0, 4.0, 6.0, 8.0, 10.0]));
      expect(result.length, equals(5));

      vector3 = new TypedVector.from([1.0, 2.0, 3.0, 4.0, 5.0]);
      vector4 = new TypedVector.from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

      expect(() => vector3 + vector4, throwsRangeError);
    });

    test('Vectors subtraction: ', () {
      TypedVector result = vector1 - vector2;

      expect(result, equals([0.0, 0.0, 0.0, 0.0, 0.0]));
      expect(result.length, equals(5));

      vector3 = new TypedVector.from([1.0, 2.0, 3.0, 4.0, 5.0]);
      vector4 = new TypedVector.from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

      expect(() => vector3 - vector4, throwsRangeError);
    });

    test('Vectors multiplication: ', () {
      TypedVector result = vector1 * vector2;

      expect(result, equals([1.0, 4.0, 9.0, 16.0, 25.0]));
      expect(result.length, equals(5));

      vector3 = new TypedVector.from([1.0, 2.0, 3.0, 4.0, 5.0]);
      vector4 = new TypedVector.from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

      expect(() => vector3 * vector4, throwsRangeError);
    });

    test('Vectors division: ', () {
      TypedVector result = vector1 / vector2;

      expect(result, equals([1.0, 1.0, 1.0, 1.0, 1.0]));
      expect(result.length, equals(5));

      vector3 = new TypedVector.from([1.0, 2.0, 3.0, 4.0, 5.0]);
      vector4 = new TypedVector.from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

      expect(() => vector3 / vector4, throwsRangeError);
    });

    test('element-wise vector power:\n', () {
      TypedVector result = vector1.intPow(3);

      expect(result != vector1, isTrue);
      expect(result.length, equals(5));
      expect(result, equals([1.0, 8.0, 27.0, 64.0, 125.0]));
      expect(() => result[5], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('element-wise vector power, inplace:\n', () {
      TypedVector result = vector1.intPow(3, inPlace: true);

      expect(result, same(vector1));
      expect(vector1.length, equals(5));
      expect(vector1, equals([1.0, 8.0, 27.0, 64.0, 125.0]));
      expect(() => vector1[5], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
    });

    test('vector multiplication (scalar format):\n', () {
      double result = vector1.vectorScalarMult(vector2);

      expect(result, equals(55.0));
    });

    test('vector and scalar multiplication:\n', () {
      TypedVector result = vector1.scalarMult(2.0);

      expect(result != vector1, isTrue);
      expect(result.length, equals(5));
      expect(result, equals([2.0, 4.0, 6.0, 8.0, 10.0]));
      expect(() => result[5], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('vector and scalar multiplication, inplace:\n', () {
      TypedVector result = vector1.scalarMult(2.0, inPlace: true);

      expect(result, same(vector1));
      expect(vector1.length, equals(5));
      expect(vector1, equals([2.0, 4.0, 6.0, 8.0, 10.0]));
      expect(() => vector1[5], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
    });

    test('vector and scalar division:\n', () {
      TypedVector result = vector1.scalarDivision(2.0);

      expect(result != vector1, isTrue);
      expect(result.length, equals(5));
      expect(result, equals([0.5, 1.0, 1.5, 2.0, 2.5]));
      expect(() => result[5], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('vector and scalar division, inplace:\n', () {
      TypedVector result = vector1.scalarDivision(2.0, inPlace: true);

      expect(result, same(vector1));
      expect(vector1.length, equals(5));
      expect(vector1, equals([0.5, 1.0, 1.5, 2.0, 2.5]));
      expect(() => vector1[5], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
    });

    test('add a scalar to a vector:\n', () {
      TypedVector result = vector1.scalarAddition(13.0);

      expect(result != vector1, isTrue);
      expect(result.length, equals(5));
      expect(result, equals([14.0, 15.0, 16.0, 17.0, 18.0]));
      expect(() => result[5], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('add a scalar to a vector, inplace:\n', () {
      TypedVector result = vector1.scalarAddition(13.0, inPlace: true);

      expect(result, same(vector1));
      expect(vector1.length, equals(5));
      expect(vector1, equals([14.0, 15.0, 16.0, 17.0, 18.0]));
      expect(() => vector1[5], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
    });

    test('subtract a scalar from a vector:\n', () {
      TypedVector result = vector1.scalarSubtraction(13.0);

      expect(result != vector1, isTrue);
      expect(result.length, equals(5));
      expect(result, equals([-12.0, -11.0, -10.0, -9.0, -8.0]));
      expect(() => result[5], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('subtract a scalar from a vector, inplace:\n', () {
      TypedVector result = vector1.scalarSubtraction(13.0, inPlace: true);

      expect(result, same(vector1));
      expect(vector1.length, equals(5));
      expect(vector1, equals([-12.0, -11.0, -10.0, -9.0, -8.0]));
      expect(() => vector1[5], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
    });

    test('find the euclidean distance between two identical vectors:\n', () {
      double distance1 = vector1.distanceTo(vector2);

      expect(distance1, equals(0.0), reason: 'Wrong vector distance calculation');
    });

    test('find the euclidean distance between two different vectors:\n', () {
      vector1 = new TypedVector.from([10.0, 3.0, 4.0, 7.0, 9.0, 12.0]);
      vector2 = new TypedVector.from([1.0, 3.0, 2.0, 11.5, 10.0, 15.5]);
      double distance2 = vector1.distanceTo(vector2);

      expect(distance2, equals(10.88577052853862), reason: 'Wrong vector distance calculation');
    });

    test('find the euclidean norm of a vector', () {
      expect(vector1.norm(), equals(7.416198487095663), reason: 'Wrong norm calculation');
    });
  });

  group('Common list operations.', () {
    setUp(() {
      vector1 = new TypedVector.from([1.0, 3.0, 2.0, 11.5]);
      vector2 = new TypedVector.from([1.0, 3.0, 2.0, 11.5, 10.0]);
      vector3 = new TypedVector.from([1.0, 3.0, 2.0, 11.5, 10.0, 15.5]);
      vector4 = new TypedVector.from([1.0, 3.0, 2.0, 11.5, 10.0, 15.5, 17.5]);
    });

    tearDown(() {
      vector1 = null;
      vector2 = null;
      vector3 = null;
      vector4 = null;
    });

    test('Add value to the end of a vector (offset = 0)', () {
      vector1.add(3.0);

      expect(vector1.length, equals(5));
      expect(vector1[4], equals(3.0));
      expect(vector1, equals([1.0, 3.0, 2.0, 11.5, 3.0]));
    });

    test('Add value to the end of a vector (offset = 3)', () {
      vector2.add(3.0);

      expect(vector2.length, equals(6));
      expect(vector2[5], equals(3.0));
      expect(vector2, equals([1.0, 3.0, 2.0, 11.5, 10.0, 3.0]));
    });

    test('Add value to the end of a vector (offset = 2)', () {
      vector3.add(3.0);

      expect(vector3.length, equals(7));
      expect(vector3[6], equals(3.0));
      expect(vector3, equals([1.0, 3.0, 2.0, 11.5, 10.0, 15.5, 3.0]));
    });

    test('Add value to the end of a vector (offset = 1)', () {
      vector4.add(3.0);

      expect(vector4.length, equals(8));
      expect(vector4[7], equals(3.0));
      expect(vector4, equals([1.0, 3.0, 2.0, 11.5, 10.0, 15.5, 17.5, 3.0]));
    });

    test('`Foreach` method testing', () {
      List<double> testList = [];

      vector3.forEach((double value) {
        testList.add(value);
      });

      expect(testList, equals(vector3));
    });
  });
}
