import 'dart:math' as math;
import 'dart:collection';

import 'package:dart_ml/src/enums.dart';
import 'package:dart_ml/src/math/vector_interface.dart';

class RegularVector extends ListBase<double> implements VectorInterface {
  final List<double> _innerList;

  RegularVector.fromList(List<double> this._innerList);

  void set length(int value) {_innerList.length = value;}
  int get length => _innerList.length;

  double operator [] (int index) => _innerList[index];
  void operator []= (int index, double value) {_innerList[index] = value;}

  RegularVector operator + (VectorInterface vector) => _elementWiseOperation(vector, (a,b) => a + b, false);
  RegularVector operator - (VectorInterface vector) => _elementWiseOperation(vector, (a,b) => a - b, false);
  RegularVector operator * (VectorInterface vector) => _elementWiseOperation(vector, (a,b) => a * b, false);
  RegularVector operator / (VectorInterface vector) => _elementWiseOperation(vector, (a,b) => a / b, false);

  RegularVector pow(double degree, {bool inPlace = false}) => _elementWiseOperation(degree, (a,b) => math.pow(a, b), inPlace);
  RegularVector scalarAddition(double value, {bool inPlace = false}) => _elementWiseOperation(value, (a,b) => a + b, inPlace);
  RegularVector scalarSubtraction(double value, {bool inPlace = false}) => _elementWiseOperation(value, (a,b) => a - b, inPlace);
  RegularVector scalarMult(double value, {bool inPlace = false}) => _elementWiseOperation(value, (a,b) => a * b, inPlace);
  RegularVector scalarDivision(double value, {bool inPlace = false}) => _elementWiseOperation(value, (a,b) => a / b, inPlace);

  double vectorScalarMult(VectorInterface vector) => (this * vector)._sum();
  double distanceTo(VectorInterface vector, [Norm norm = Norm.EUCLIDEAN]) => (this - vector).norm(norm);

  double norm([Norm norm = Norm.EUCLIDEAN]) {
    double degree;

    switch(norm) {
      case Norm.EUCLIDEAN:
        degree = 2.0;
        break;
    }

    return math.pow(pow(degree)._sum(), 1 / degree);
  }

  double _sum() => this.reduce((double item, double sum) => item + sum);

  RegularVector _elementWiseOperation(Object value, operation(double a, double b), bool inPlace) {
    List<double> _bufList = inPlace ? _innerList : new List<double>(this.length);

    for (int i = 0; i < this.length; i++) {
      _bufList[i] = operation(this[i], (value is RegularVector ? value[i] : value));
    }

    return inPlace ? this : new RegularVector.fromList(_bufList);
  }
}