// Copyright (c) 2017, Ilya Gyrdymov. All rights reserved. Use of this source code
// is governed by a BSD-style license that can be found in the LICENSE file.

import 'dart:typed_data';
import 'package:dart_ml/dart_ml.dart' show TypedVector;
import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

void main() {
  TypedVector vector1;
  TypedVector vector2;

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

      expect(vector1[0], equals(1.0));
      expect(vector1[4], equals(5.0));
      expect(vector1[5], equals(6.0));
      expect(vector1[11], equals(0.0));
      expect(() => vector1[12], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
      expect(vector1.length, equals(12));

      vector1 = new TypedVector.fromTypedList(typedList, 10);

      expect(vector1[0], equals(1.0));
      expect(vector1[4], equals(5.0));
      expect(vector1[5], equals(6.0));
      expect(vector1[9], equals(10.0));
      expect(() => vector1[10], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);
      expect(vector1.length, equals(10));
    });

    test('Vector initialization via `fill` constructor: ', () {
      vector1 = new TypedVector.filled(10, 2.0);

      expect(vector1[0], equals(2.0));
      expect(vector1[4], equals(2.0));
      expect(vector1[5], equals(2.0));
      expect(() => vector1[11], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);

      expect(vector1.length, equals(10));
    });

    test('Vector elements updating: ', () {
      vector1 = new TypedVector.from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

      vector1[0] = 34.0;
      vector1[4] = 277.0;
      vector1[5] = 1000.0;

      expect(vector1[0], equals(34.0));
      expect(vector1[4], equals(277.0));
      expect(vector1[5], equals(1000.0));
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
    });

    test('Vector addition: ', () {
      TypedVector result = vector1 + vector2;

      expect(result[1], equals(4.0));
      expect(result[4], equals(10.0));
      expect(result.length, equals(5));

      TypedVector vector3 = new TypedVector.from([1.0, 2.0, 3.0, 4.0, 5.0]);
      TypedVector vector4 = new TypedVector.from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

      expect(() => vector3 + vector4, throwsRangeError);
    });
  });
}
