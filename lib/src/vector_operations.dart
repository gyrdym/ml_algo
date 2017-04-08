import 'dart:math' as math;
import 'dart:collection';
import 'dart:typed_data';

import 'package:dart_ml/src/enums.dart' show Norm;

class Vector extends ListBase {
  Float32x4List _innerList;
  int _origLength;

  Vector.from(List<double> source) {
    _origLength = source.length;
    _innerList = _convertRegularListToTyped(source);
  }

  ///Very slow operation
  void set length(int newLength) {
    if (newLength == _origLength) {
      return;
    }

    _origLength = newLength;

    if (newLength == _innerList.length * 4) {
      return;
    }

    List<double> _buffList = _convertTypedListToRegular(_innerList);
    _buffList.length = newLength;
    _innerList = _convertRegularListToTyped(_buffList);
  }

  int get length => _origLength;

  double operator [](int index) {
    if (index > (_origLength - 1) || index < 0) {
      throw new RangeError("Index out of range!");
    }

    int base = (index / 4).floor();
    int offset = index - (base * 4);

    switch (offset) {
      case 0:
        return _innerList[base].x;
      case 1:
        return _innerList[base].y;
      case 2:
        return _innerList[base].z;
      case 3:
        return _innerList[base].w;
      default:
        throw new RangeError("Index out of range!");
    }
  }

  void operator []=(int index, double value) {
    if (index > (_origLength - 1) || index < 0) {
      throw new RangeError("Index out of range!");
    }

    int base = (index / 4).floor();
    int offset = index - (base * 4);

    switch (offset) {
      case 0:
        _innerList[base] = _innerList[base].withX(value);
        break;
      case 1:
        _innerList[base] = _innerList[base].withY(value);
        break;
      case 2:
        _innerList[base] = _innerList[base].withZ(value);
        break;
      case 3:
        _innerList[base] = _innerList[base].withW(value);
        break;
      default:
        throw new RangeError("Index out of range!");
    }
  }

  Float32x4List _convertRegularListToTyped(List<double> source) {
    int partsCount = (_origLength / 4).ceil();
    List<Float32x4>_bufferList = [];

    for (int i = 0; i < partsCount; i++) {
      int end = (i + 1) * 4;
      int start = end - 4;
      int diff = end - _origLength;

      if (diff > 0) {
        source.addAll(new List.filled(diff, 0.0));
      }

      List<double> sublist = source.sublist(start, end);
      double x = sublist[0] ?? 0.0;
      double y = sublist[1] ?? 0.0;
      double z = sublist[2] ?? 0.0;
      double w = sublist[3] ?? 0.0;

      _bufferList.add(new Float32x4(x, y, z, w));
    }

    return new Float32x4List.fromList(_bufferList);
  }

  List<double> _convertTypedListToRegular(Float32x4List source) {
    List<double> _buffList = [];

    for (int i = 0; i < _innerList.length; i++) {
      Float32x4 item = _innerList[i];
      _buffList.addAll([item.x, item.y, item.z, item.w]);
    }

    return _buffList;
  }
}

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

List<double> add(List<double> a, double b) {
  List<double> result = new List<double>();

  for (int i = 0; i < a.length; i++) {
    result.add(a[i] + b);
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

List<double> mult(List<double> a, List<double> b) {
  if (a.length != b.length) {
    throw new Exception('Lists must have the same length!');
  }

  List<double> vector = new List<double>();

  for (int i = 0; i < a.length; i++) {
    vector.add(a[i] * b[i]);
  }

  return vector;
}

List<double> create(int len, [double initialValue = 0.0]) {
  List<double> vector = new List<double>();

  for (int i = 0; i < len; i++) {
    vector.add(initialValue);
  }

  return vector;
}
