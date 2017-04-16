import 'dart:math' as math;
import 'dart:collection';

import 'package:dart_ml/src/enums.dart';
import 'package:dart_ml/src/math/vector_interface.dart';

class RegularVector extends ListBase<double> implements VectorInterface {
  List<double> _innerList;

  RegularVector(int dimension) {
    _innerList = new List<double>(dimension);
  }

  RegularVector.from(List<double> source) {
    _innerList = source.toList(growable: false);
  }

  RegularVector.filled(int dimension, double value) {
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

  double vectorScalarMult(VectorInterface vector) => (this * vector)._sum();
  double distanceTo(VectorInterface vector, [Norm norm = Norm.EUCLIDEAN]) => (this - vector).norm(norm);

  double norm([Norm norm = Norm.EUCLIDEAN]) {
    int exponent;

    switch(norm) {
      case Norm.EUCLIDEAN:
        exponent = 2;
        break;
    }

    return math.pow(intPow(exponent)._sum(), 1 / exponent);
  }

  double mean() => _sum() / length;

  void add(double value) {
    _innerList = _innerList.toList(growable: true)..add(value);
    _innerList = _innerList.toList(growable: false);
  }

  double _sum() => this._innerList.reduce((double item, double sum) => item + sum);

  RegularVector _elementWiseOperation(Object value, operation(double a, double b), bool inPlace) {
    List<double> _bufList = inPlace ? _innerList : new List<double>(this.length);

    for (int i = 0; i < this.length; i++) {
      _bufList[i] = operation(this[i], (value is RegularVector ? value[i] : value));
    }

    return inPlace ? this : new RegularVector.from(_bufList);
  }
}