import 'dart:math' as math;

import 'package:dart_ml/src/enums.dart' show Norm;

double distance(List<double> a, List<double> b, [Norm norm = Norm.EUCLIDEAN]) {
  if (a.length != b.length) {
    throw new Exception('Lists must have the same length!');
  }

  double sum = 0.0;
  int pow;

  switch (norm) {
    case Norm.EUCLIDEAN:
      pow = 2;
  }

  for (int i = 0; i < a.length; i ++) {
    sum += math.pow(a[i] - b[i], pow);
  }

  return math.sqrt(sum);
}

List<double> subtraction(List<double> a, List<double> b) {
  if (a.length != b.length) {
    throw new Exception('Lists must have the same length!');
  }

  List<double> result = new List<double>();

  for (int i = 0; i < a.length; i ++) {
    result.add(a[i] - b[i]);
  }

  return result;
}

List<double> pow(List<double> a, num exponent) {
  List<double> result = new List<double>();

  for (int i = 0; i < a.length; i++) {
    result.add(math.pow(a[i], exponent));
  }

  return result;
}

double mean(List<double> a) {
  double sum = 0.0;

  for (int i = 0; i < a.length; i++) {
    sum += a[i];
  }

  return sum / a.length;
}

double scalarMult(List<double> a, List<double> b) {
  if (a.length != b.length) {
    throw new Exception('Lists must have the same length!');
  }

  double sum = 0.0;

  for (int i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }

  return sum;
}

List<double> create(int len, [double initialValue = 0.0]) {
  List<double> vector = new List<double>();

  for (int i = 0; i < len; i++) {
    vector.add(initialValue);
  }

  return vector;
}
