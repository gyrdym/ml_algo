import 'dart:math' as math;

import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/math/vector/vector_abstract.dart';

class RegularVector extends Vector {
  List<double> _innerList;

  RegularVector(int dimension) {
    _innerList = new List<double>(dimension);
  }

  RegularVector.from(Iterable<double> source) : super.from(source) {
    _innerList = source.toList(growable: false);
  }

  RegularVector.filled(int dimension, double value) : super.filled(dimension, value) {
    _innerList = new List<double>.filled(dimension, value, growable: false);
  }

  ///it's a high-cost operation, cause dimension changing means fully vector re-creation
  void set length(int value) {
    if (value == length) {
      return;
    }

    if (value < length) {
      _innerList = _innerList.take(value).toList(growable: false);
      return;
    }

    _innerList = _innerList.toList(growable: true)
      ..length = value
      ..fillRange(length, value, 0.0);

    _innerList = _innerList.toList(growable: false);
  }

  int get length => _innerList.length;

  double operator [] (int index) => _innerList[index];
  void operator []= (int index, double value) {_innerList[index] = value;}

  RegularVector operator + (VectorInterface vector) => _elementWiseOperation(vector, (a,b) => a + b, false);
  RegularVector operator - (VectorInterface vector) => _elementWiseOperation(vector, (a,b) => a - b, false);
  RegularVector operator * (VectorInterface vector) => _elementWiseOperation(vector, (a,b) => a * b, false);
  RegularVector operator / (VectorInterface vector) => _elementWiseOperation(vector, (a,b) => a / b, false);

  RegularVector intPow(int exponent, {bool inPlace = false}) => _elementWiseOperation(exponent, (a,b) => math.pow(a, b), inPlace);
  RegularVector scalarAddition(double value, {bool inPlace = false}) => _elementWiseOperation(value, (a,b) => a + b, inPlace);
  RegularVector scalarSubtraction(double value, {bool inPlace = false}) => _elementWiseOperation(value, (a,b) => a - b, inPlace);
  RegularVector scalarMult(double value, {bool inPlace = false}) => _elementWiseOperation(value, (a,b) => a * b, inPlace);
  RegularVector scalarDivision(double value, {bool inPlace = false}) => _elementWiseOperation(value, (a,b) => a / b, inPlace);

  void add(double value) {
    _innerList = _innerList.toList(growable: true)..add(value);
    _innerList = _innerList.toList(growable: false);
  }

  RegularVector fromRange(int start, [int end]) => new RegularVector.from(sublist(start, end));
  RegularVector copy() => fromRange(0);

  void fill(double value) {
    _innerList.fillRange(0, _innerList.length, value);
  }

  double sum() => this._innerList.reduce((double item, double sum) => item + sum);

  RegularVector abs({bool inPlace = false}) {
    List<double> list = _innerList.map((double item) => item.abs()).toList(growable: false);

    if (inPlace) {
      _innerList = list;
      return this;
    }

    return new RegularVector.from(list);
  }

  RegularVector _elementWiseOperation(Object value, operation(double a, double b), bool inPlace) {
    List<double> _bufList = inPlace ? _innerList : new List<double>(this.length);

    for (int i = 0; i < this.length; i++) {
      _bufList[i] = operation(this[i], (value is RegularVector ? value[i] : value));
    }

    return inPlace ? this : new RegularVector.from(_bufList);
  }
}
