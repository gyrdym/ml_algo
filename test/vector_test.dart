// Copyright (c) 2017, Ilya Gyrdymov. All rights reserved. Use of this source code
// is governed by a BSD-style license that can be found in the LICENSE file.
import 'package:dart_ml/dart_ml.dart';
import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

void main() {
  Vector vector1;
  Vector vector2;

  group('Vector fundamental', () {
    setUp(() {
      vector1 = new Vector.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    });

    test('Vector initialization: ', () {
      expect(vector1[0], equals(1.0));
      expect(vector1[4], equals(5.0));
      expect(vector1[5], equals(6.0));
      expect(() => vector1[7], throwsRangeError);
      expect(() => vector1[-1], throwsRangeError);

      expect(vector1.length, equals(6));
    });

    test('Vector elements updating: ', () {
      vector1[0] = 34.0;
      vector1[4] = 277.0;
      vector1[5] = 1000.0;

      expect(vector1[0], equals(34.0));
      expect(vector1[4], equals(277.0));
      expect(vector1[5], equals(1000.0));
    });

    test('Vector length updating: ', () {
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
      vector1 = new Vector.fromList([1.0, 2.0, 3.0, 4.0, 5.0]);
      vector2 = new Vector.fromList([1.0, 2.0, 3.0, 4.0, 5.0]);
    });

    test('Vector addition: ', () {
      Vector result = vector1 + vector2;

      expect(result[1], equals(4.0));
      expect(result[4], equals(10.0));
      expect(result.length, equals(5));

      Vector vector3 = new Vector.fromList([1.0, 2.0, 3.0, 4.0, 5.0]);
      Vector vector4 = new Vector.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

      expect(() => vector3 + vector4, throwsRangeError);
    });
  });
}
