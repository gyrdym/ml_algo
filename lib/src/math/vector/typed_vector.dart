import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/math/vector/vector_abstract.dart';

class TypedVector extends Vector {
  Float32x4List _innerList;
  int _origLength;

  TypedVector(int length) {
    _innerList = new Float32x4List((length / 4).ceil());
    _origLength = length;
  }

  TypedVector.from(Iterable<double> source) {
    _origLength = source.length;
    _innerList = _convertRegularListToTyped(source);
  }

  TypedVector.fromTypedList(Float32x4List source, [int origLength]) {
    _origLength = origLength ?? source.length * 4;
    _innerList = source;
  }

  TypedVector.filled(int length, double value) {
    _origLength = length;
    _innerList = _convertRegularListToTyped(new List<double>.filled(length, value));
  }

  TypedVector.randomFilled(int length, {int seed}) {
    math.Random _randomizer = new math.Random(seed);
    List<double> _list = new List<double>.generate(length, (_) => _randomizer.nextDouble());
    _origLength = length;
    _innerList = _convertRegularListToTyped(_list);
  }

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

  TypedVector operator + (VectorInterface vector) => _elementWiseOperation(vector, (a, b) => a + b, false);
  TypedVector operator - (VectorInterface vector) => _elementWiseOperation(vector, (a, b) => a - b, false);
  TypedVector operator * (VectorInterface vector) => _elementWiseOperation(vector, (a, b) => a * b, false);
  TypedVector operator / (VectorInterface vector) => _elementWiseOperation(vector, (a, b) => a / b, false);

  TypedVector intPow(int exponent, {bool inPlace = false}) => _elementWisePow(exponent, inPlace);
  TypedVector scalarMult(double value, {bool inPlace = false}) => _elementWiseOperation(value, (a, b) => a * b, inPlace);
  TypedVector scalarDivision(double value, {bool inPlace = false}) => _elementWiseOperation(value, (a, b) => a / b, inPlace);
  TypedVector scalarAddition(double value, {bool inPlace = false}) => _elementWiseOperation(value, (a, b) => a + b, inPlace);
  TypedVector scalarSubtraction(double value, {bool inPlace = false}) => _elementWiseOperation(value, (a, b) => a - b, inPlace);

  void add(double value) {
    int diff = _innerList.length * 4 - _origLength;

    if (diff == 0) {
      _add(new Float32x4.splat(value));
    } else {
      int base = _innerList.length - 1;
      switch (diff) {
        case 1:
          _innerList[base] = _innerList.last.withW(value);
          break;
        case 2:
          _innerList[base] = _innerList.last.withZ(value);
          break;
        case 3:
          _innerList[base] = _innerList.last.withY(value);
          break;
        case 4:
          _innerList[base] = _innerList.last.withX(value);
          break;
      }
    }

    _origLength++;
  }

  double sum() {
    Float32x4 sum = this._innerList.reduce((Float32x4 item, Float32x4 sum) => item + sum);
    return sum.x + sum.y + sum.z + sum.w;
  }

  TypedVector abs({bool inPlace = false}) {
    Float32x4List list = new Float32x4List.fromList(
        _innerList.map((Float32x4 item) => item.abs())
            .toList(growable: false));

    if (inPlace) {
      _innerList = list;
      return this;
    }

    return new TypedVector.fromTypedList(list, _origLength);
  }

  TypedVector cut(int start, [int end]) => new TypedVector.from(sublist(start, end));
  TypedVector copy() => cut(0);
  TypedVector createFrom(Iterable<double> iterable) => new TypedVector.from(iterable);

  void fill(double value) {
    _innerList.fillRange(0, _innerList.length, new Float32x4.splat(value));
  }

  void concat(VectorInterface vector) {
    _concat(vector);
  }

  void _concat(TypedVector vector) {
    _innerList = new Float32x4List.fromList(_innerList.toList(growable: true)..addAll(vector._innerList));
    _origLength += vector.length;
  }

  void _add(Float32x4 value) {
    _innerList = new Float32x4List.fromList(_innerList.toList(growable: true)..add(value));
  }

  Float32x4 _laneIntPow(Float32x4 lane, int e) {
    if (e == 0) {
      return new Float32x4.splat(1.0);
    }

    Float32x4 x = _laneIntPow(lane, e ~/ 2);
    Float32x4 sqrX = x * x;

    if (e % 2 == 0) {
      return sqrX;
    }

    return (lane * sqrX);
  }

  Float32x4List _convertRegularListToTyped(List<double> source) {
    int partsCount = (source.length / 4).ceil();
    Float32x4List _bufferList = new Float32x4List(partsCount);

    for (int i = 0; i < partsCount; i++) {
      int end = (i + 1) * 4;
      int start = end - 4;
      int diff = end - source.length;
      List<double> sublist;

      if (diff > 0) {
        List<double> zeroItems = new List<double>.filled(diff, 0.0);
        sublist = source.sublist(start);
        sublist.addAll(zeroItems);
      } else {
        sublist = source.sublist(start, end);
      }

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

    for (int i = 0; i < source.length; i++) {
      Float32x4 item = source[i];
      _buffList.addAll([item.x, item.y, item.z, item.w]);
    }

    return _buffList;
  }

  TypedVector _elementWiseOperation(Object value, operation(Float32x4 a, Float32x4 b), bool inPlace) {
    if (value is TypedVector && value.length != this.length) {
      throw _mismatchLengthError();
    }

    Float32x4 _typedValue;

    if (value is! TypedVector) {
      _typedValue = new Float32x4.splat(value);
    }

    Float32x4List _bufList = inPlace ? _innerList : new Float32x4List(this._innerList.length);

    for (int i = 0; i < this._innerList.length; i++) {
      _bufList[i] = operation(this._innerList[i], value is TypedVector ? value._innerList[i] : _typedValue);
    }

    return inPlace ? this : new TypedVector.fromTypedList(_bufList, _origLength);
  }

  TypedVector _elementWisePow(int exp, bool inPlace) {
    Float32x4List _bufList = inPlace ? _innerList : new Float32x4List(_innerList.length);

    for (int i = 0; i < _innerList.length; i++) {
      _bufList[i] = _laneIntPow(_innerList[i], exp);
    }

    return inPlace ? this : new TypedVector.fromTypedList(_bufList, _origLength);
  }

  RangeError _outOfRangeError(index) => new RangeError("Index $index out of range!");
  RangeError _lengthRangeError(value) => new RangeError.value(value, 'length', 'Invalid length: length must be positive'
      ' or equal to zero');
  RangeError _mismatchLengthError() => new RangeError('Vectors length must be equal');
}
