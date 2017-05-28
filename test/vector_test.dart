import 'dart:typed_data';
import 'package:dart_ml/dart_ml.dart' show Vector, Norm;
import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

void main() {
  Vector vector1;
  Vector vector2;
  Vector vector3;
  Vector vector4;

  group('Vector fundamental:\n', () {
    tearDown(() {
      vector1 = null;
      vector2 = null;
    });

    test('Vector initialization via default constructor... ', () {
      vector1 = new Vector(5);

      expect(vector1, equals([0.0, 0.0, 0.0, 0.0, 0.0]));
      expect(() => vector1[11], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);

      expect(vector1.length, equals(5));
    });

    test('Vector initialization via `from` constructor... ', () {
      //dynamic-length list
      vector1 = new Vector.from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

      expect(vector1[0], equals(1.0));
      expect(vector1[4], equals(5.0));
      expect(vector1[5], equals(6.0));
      expect(() => vector1[7], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);

      expect(vector1.length, equals(6));

      //fixed-length list
      vector2 = new Vector.from(new List.filled(11, 1.0));

      expect(vector2.length, 11);
      expect(vector2[10], 1.0);
    });

    test('Vector initialization via `fromTypedList` constructor... ', () {
      Float32x4List typedList = new Float32x4List.fromList([
        new Float32x4(1.0, 2.0, 3.0, 4.0),
        new Float32x4(5.0, 6.0, 7.0, 8.0),
        new Float32x4(9.0, 10.0, 0.0, 0.0)
      ]);

      vector1 = new Vector.fromTypedList(typedList);

      expect(vector1, equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0]));
      expect(() => vector1[12], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
      expect(vector1.length, equals(12));

      vector1 = new Vector.fromTypedList(typedList, 10);

      expect(vector1, equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]));
      expect(() => vector1[10], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
      expect(vector1.length, equals(10));
    });

    test('Vector initialization via `fill` constructor... ', () {
      vector1 = new Vector.filled(10, 2.0);

      expect(vector1, equals([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]));
      expect(() => vector1[11], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);

      expect(vector1.length, equals(10));
    });

    test('Vector elements updating... ', () {
      vector1 = new Vector.from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

      vector1[0] = 34.0;
      vector1[4] = 277.0;
      vector1[5] = 1000.0;

      expect(vector1, equals([34.0, 2.0, 3.0, 4.0, 277.0, 1000.0]));
      expect(vector1.length, equals(6));
    });

    test('Vector length updating... ', () {
      vector1 = new Vector.from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

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

  group('Vector operations:\n', () {
    setUp(() {
      vector1 = new Vector.from([1.0, 2.0, 3.0, 4.0, 5.0]);
      vector2 = new Vector.from([1.0, 2.0, 3.0, 4.0, 5.0]);
    });

    tearDown(() {
      vector1 = null;
      vector2 = null;
      vector3 = null;
      vector4 = null;
    });

    test('Vectors addition... ', () {
      Vector result = vector1 + vector2;

      expect(result, equals([2.0, 4.0, 6.0, 8.0, 10.0]));
      expect(result.length, equals(5));

      vector3 = new Vector.from([1.0, 2.0, 3.0, 4.0, 5.0]);
      vector4 = new Vector.from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

      expect(() => vector3 + vector4, throwsRangeError);
    });

    test('Vectors subtraction... ', () {
      Vector result = vector1 - vector2;

      expect(result, equals([0.0, 0.0, 0.0, 0.0, 0.0]));
      expect(result.length, equals(5));

      vector3 = new Vector.from([1.0, 2.0, 3.0, 4.0, 5.0]);
      vector4 = new Vector.from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

      expect(() => vector3 - vector4, throwsRangeError);
    });

    test('Vectors multiplication... ', () {
      Vector result = vector1 * vector2;

      expect(result, equals([1.0, 4.0, 9.0, 16.0, 25.0]));
      expect(result.length, equals(5));

      vector3 = new Vector.from([1.0, 2.0, 3.0, 4.0, 5.0]);
      vector4 = new Vector.from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

      expect(() => vector3 * vector4, throwsRangeError);
    });

    test('Vectors division... ', () {
      Vector result = vector1 / vector2;

      expect(result, equals([1.0, 1.0, 1.0, 1.0, 1.0]));
      expect(result.length, equals(5));

      vector3 = new Vector.from([1.0, 2.0, 3.0, 4.0, 5.0]);
      vector4 = new Vector.from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

      expect(() => vector3 / vector4, throwsRangeError);
    });

    test('element-wise vector power... ', () {
      Vector result = vector1.intPow(3);

      expect(result != vector1, isTrue);
      expect(result.length, equals(5));
      expect(result, equals([1.0, 8.0, 27.0, 64.0, 125.0]));
      expect(() => result[5], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('element-wise vector power, inplace... ', () {
      Vector result = vector1.intPow(3, inPlace: true);

      expect(result, same(vector1));
      expect(vector1.length, equals(5));
      expect(vector1, equals([1.0, 8.0, 27.0, 64.0, 125.0]));
      expect(() => vector1[5], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
    });

    test('vector multiplication (scalar format)... ', () {
      double result = vector1.dot(vector2);

      expect(result, equals(55.0));
    });

    test('vector and scalar multiplication... ', () {
      Vector result = vector1.scalarMul(2.0);

      expect(result != vector1, isTrue);
      expect(result.length, equals(5));
      expect(result, equals([2.0, 4.0, 6.0, 8.0, 10.0]));
      expect(() => result[5], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('vector and scalar multiplication, inplace... ', () {
      Vector result = vector1.scalarMul(2.0, inPlace: true);

      expect(result, same(vector1));
      expect(vector1.length, equals(5));
      expect(vector1, equals([2.0, 4.0, 6.0, 8.0, 10.0]));
      expect(() => vector1[5], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
    });

    test('vector and scalar division... ', () {
      Vector result = vector1.scalarDiv(2.0);

      expect(result != vector1, isTrue);
      expect(result.length, equals(5));
      expect(result, equals([0.5, 1.0, 1.5, 2.0, 2.5]));
      expect(() => result[5], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('vector and scalar division, inplace... ', () {
      Vector result = vector1.scalarDiv(2.0, inPlace: true);

      expect(result, same(vector1));
      expect(vector1.length, equals(5));
      expect(vector1, equals([0.5, 1.0, 1.5, 2.0, 2.5]));
      expect(() => vector1[5], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
    });

    test('add a scalar to a vector... ', () {
      Vector result = vector1.scalarAdd(13.0);

      expect(result != vector1, isTrue);
      expect(result.length, equals(5));
      expect(result, equals([14.0, 15.0, 16.0, 17.0, 18.0]));
      expect(() => result[5], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('add a scalar to a vector, inplace... ', () {
      Vector result = vector1.scalarAdd(13.0, inPlace: true);

      expect(result, same(vector1));
      expect(vector1.length, equals(5));
      expect(vector1, equals([14.0, 15.0, 16.0, 17.0, 18.0]));
      expect(() => vector1[5], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
    });

    test('subtract a scalar from a vector... ', () {
      Vector result = vector1.scalarSub(13.0);

      expect(result != vector1, isTrue);
      expect(result.length, equals(5));
      expect(result, equals([-12.0, -11.0, -10.0, -9.0, -8.0]));
      expect(() => result[5], throwsRangeError);
      expect(() => result[-1], throwsRangeError);
    });

    test('subtract a scalar from a vector, inplace... ', () {
      Vector result = vector1.scalarSub(13.0, inPlace: true);

      expect(result, same(vector1));
      expect(vector1.length, equals(5));
      expect(vector1, equals([-12.0, -11.0, -10.0, -9.0, -8.0]));
      expect(() => vector1[5], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
    });

    test('find the euclidean distance between two identical vectors... ', () {
      double distance1 = vector1.distanceTo(vector2);

      expect(distance1, equals(0.0), reason: 'Wrong vector distance calculation');
    });

    test('find the distance between two different vectors... ', () {
      vector1 = new Vector.from([10.0, 3.0, 4.0, 7.0, 9.0, 12.0]);
      vector2 = new Vector.from([1.0, 3.0, 2.0, 11.5, 10.0, 15.5]);

      expect(vector1.distanceTo(vector2, Norm.EUCLIDEAN), equals(10.88577052853862), reason: 'Wrong vector distance calculation');
      expect(vector1.distanceTo(vector2, Norm.MANHATTAN), equals(20.0), reason: 'Wrong vector distance calculation');
    });

    test('find a norm of a vector', () {
      expect(vector1.norm(Norm.EUCLIDEAN), equals(7.416198487095663), reason: 'Wrong norm calculation');
      expect(vector1.norm(Norm.MANHATTAN), equals(15.0), reason: 'Wrong norm calculation');
    });

    test('find the sum of vector elements... ', () {
      expect(vector1.sum(), equals(15.0));
    });

    test('find the absolute value of an each element of a vector... ', () {
      vector1 = new Vector.from([-3.0, 4.5, -12.0, -23.5, 44.0]);
      Vector result = vector1.abs();
      expect(result, equals([3.0, 4.5, 12.0, 23.5, 44.0]));
      expect(result, isNot(vector1));
    });

    test('find the absolute value of an each element of a vector (in place)... ', () {
      vector1 = new Vector.from([-3.0, 4.5, -12.0, -23.5, 44.0]);
      Vector result = vector1.abs(inPlace: true);
      expect(result, equals([3.0, 4.5, 12.0, 23.5, 44.0]));
      expect(result, same(vector1));
    });
  });

  group('Common list operations:\n', () {
    setUp(() {
      vector1 = new Vector.from([1.0, 3.0, 2.0, 11.5]);
      vector2 = new Vector.from([1.0, 3.0, 2.0, 11.5, 10.0]);
      vector3 = new Vector.from([1.0, 3.0, 2.0, 11.5, 10.0, 15.5]);
      vector4 = new Vector.from([1.0, 3.0, 2.0, 11.5, 10.0, 15.5, 17.5]);
    });

    tearDown(() {
      vector1 = null;
      vector2 = null;
      vector3 = null;
      vector4 = null;
    });

    test('Add value to the end of a vector (offset = 0)... ', () {
      vector1.add(3.0);

      expect(vector1.length, equals(5));
      expect(vector1[4], equals(3.0));
      expect(vector1, equals([1.0, 3.0, 2.0, 11.5, 3.0]));
    });

    test('Add value to the end of a vector (offset = 3)... ', () {
      vector2.add(3.0);

      expect(vector2.length, equals(6));
      expect(vector2[5], equals(3.0));
      expect(vector2, equals([1.0, 3.0, 2.0, 11.5, 10.0, 3.0]));
    });

    test('Add value to the end of a vector (offset = 2)... ', () {
      vector3.add(3.0);

      expect(vector3.length, equals(7));
      expect(vector3[6], equals(3.0));
      expect(vector3, equals([1.0, 3.0, 2.0, 11.5, 10.0, 15.5, 3.0]));
    });

    test('Add value to the end of a vector (offset = 1)... ', () {
      vector4.add(3.0);

      expect(vector4.length, equals(8));
      expect(vector4[7], equals(3.0));
      expect(vector4, equals([1.0, 3.0, 2.0, 11.5, 10.0, 15.5, 17.5, 3.0]));
    });

    test('`cut` method testing... ', () {
      expect(vector4.cut(2, 4), equals([2.0, 11.5]));
      expect(vector4.cut(2, 5), equals([2.0, 11.5, 10.0]));
      expect(vector4.cut(2), equals([2.0, 11.5, 10.0, 15.5, 17.5]));
      expect(() => vector4.cut(-1), throwsRangeError);
      expect(() => vector4.cut(10), throwsRangeError);
    });

    test('`copy` method testing... ', () {
      Vector tmp = vector1.copy();
      expect(tmp, equals([1.0, 3.0, 2.0, 11.5]));
      expect(tmp == vector1, false);
    });

    test('`fill` method testing... ', () {
      vector1.fill(1.0);
      expect(vector1, equals([1.0, 1.0, 1.0, 1.0]));
    });

    test('`addAll` method testing... ', () {
      vector1.addAll([2.0 , 3.0, 4.0, 5.0, 6.0]);
      expect(vector1, equals([1.0, 3.0, 2.0, 11.5, 2.0 , 3.0, 4.0, 5.0, 6.0]));
    });

    test('`getRange` method testing... ', () {
      Iterable<double> res = vector4.getRange(0, 3);
      expect(res, equals([1.0, 3.0, 2.0]));
    });
  });
}
