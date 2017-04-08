// Copyright (c) 2017, Ilya Gyrdymov. All rights reserved. Use of this source code
// is governed by a BSD-style license that can be found in the LICENSE file.
import 'package:dart_ml/dart_ml.dart';
import 'package:test/test.dart';
import 'package:matcher/matcher.dart';

void main() {
  Vector vector;

  group('Vector fundamental', () {
    setUp(() {
      vector = new Vector.from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    });

    test('Vector initialization: ', () {
      expect(vector[0], equals(1.0));
      expect(vector[4], equals(5.0));
      expect(vector[5], equals(6.0));
      expect(() => vector[7], throwsRangeError);
      expect(() => vector[-1], throwsRangeError);

      expect(vector.length, equals(6));
    });

    test('Vector elements updating: ', () {
      vector[0] = 34.0;
      vector[4] = 277.0;
      vector[5] = 1000.0;

      expect(vector[0], equals(34.0));
      expect(vector[4], equals(277.0));
      expect(vector[5], equals(1000.0));
    });

    test('Vector length updating: ', () {
      vector.length = 8;
      expect(vector.length, 8);

      vector.length = 13;
      expect(vector.length, 13);
      expect(vector[12], equals(0.0));

      vector.length = 0;
      expect(vector.length, 0);
      expect(() => vector[1], throwsRangeError);

      expect(() => vector.length = -2, throwsRangeError);
    });
  });
}
