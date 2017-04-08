import 'dart:math' as math;
import 'dart:collection';
import 'dart:typed_data';

import 'package:dart_ml/src/enums.dart' show Norm;

class Vector extends ListBase {
  Float32x4List _innerList;
  int _origLength;

  Vector.fromList(List<double> source) {
    _origLength = source.length;
    _innerList = _convertRegularListToTyped(source);
  }

  Vector.fromTypedList(Float32x4List source, [int origLength]) {
    _origLength = origLength ?? source.length * 4;
    _innerList = source;
  }

  Float32x4List get typedList => _innerList;

  ///Very slow operation
  void set length(int newLength) {
    if (newLength < 0) {
      throw _lengthRangeError(newLength);
    }

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

  ///Do not use it in iteration! Use it only to read certain element
  double operator [](int index) {
    if (index > (_origLength - 1) || index < 0) {
      throw _outOfRangeError(index);
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
        throw _outOfRangeError(index);
    }
  }

  ///Do not use it in iteration! Use it only to update certain element
  void operator []=(int index, double value) {
    if (index > (_origLength - 1) || index < 0) {
      throw _outOfRangeError(index);
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
        throw _outOfRangeError(index);
    }
  }

  Vector operator +(Vector vector) {
    if (vector.length != this.length) {
      throw _mismatchLengthError();
    }

    Float32x4List _bufList = new Float32x4List(this.typedList.length);

    for (int i = 0; i < this.typedList.length; i++) {
      _bufList[i] = vector.typedList[i] + this.typedList[i];
    }

    return new Vector.fromTypedList(_bufList, this.length);
  }

  Float32x4List _convertRegularListToTyped(List<double> source) {
    int partsCount = (_origLength / 4).ceil();
    Float32x4List _bufferList = new Float32x4List(partsCount);

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

      _bufferList[i] = new Float32x4(x, y, z, w);
    }

    return _bufferList;
  }

  List<double> _convertTypedListToRegular(Float32x4List source) {
    List<double> _buffList = [];

    for (int i = 0; i < _innerList.length; i++) {
      Float32x4 item = _innerList[i];
      _buffList.addAll([item.x, item.y, item.z, item.w]);
    }

    return _buffList;
  }

  RangeError _outOfRangeError(index) => new RangeError("Index $index out of range!");
  RangeError _lengthRangeError(value) => new RangeError.value(value, 'length', 'Invalid length: length must be positive'
      ' or equal to zero');
  RangeError _mismatchLengthError() => new RangeError('Vectors length must be equal');
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
