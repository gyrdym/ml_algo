import 'dart:math' as math;
import 'dart:collection';

import 'package:dart_ml/src/enums.dart';
import 'package:dart_ml/src/math/vector_interface.dart';

class RegularVector extends ListBase<double> implements VectorInterface {
  List<double> _innerList;

  RegularVector.fromList(List<double> source) {
    _innerList = source;
  }

  double operator [] (int index) => _innerList[index];

  void operator []= (int index, double value) {
    _innerList[index] = value;
  }

  RegularVector operator + (VectorInterface vector) {
    List<double> _bufList = new List<double>(this.length);

    for (int i = 0; i < this.length; i++) {
      _bufList[i] = this[i] + vector[i];
    }

    return new RegularVector.fromList(_bufList);
  }

  RegularVector operator - (VectorInterface vector) {
    List<double> _bufList = new List<double>(this.length);

    for (int i = 0; i < this.length; i++) {
      _bufList[i] = this[i] - vector[i];
    }

    return new RegularVector.fromList(_bufList);
  }

  RegularVector operator * (VectorInterface vector) {
    List<double> _bufList = new List<double>(this.length);

    for (int i = 0; i < this.length; i++) {
      _bufList[i] = this[i] * vector[i];
    }

    return new RegularVector.fromList(_bufList);
  }

  RegularVector operator / (VectorInterface vector) {
    List<double> _bufList = new List<double>(this.length);

    for (int i = 0; i < this.length; i++) {
      _bufList[i] = this[i] / vector[i];
    }

    return new RegularVector.fromList(_bufList);
  }

  void set length(int value) {
    _innerList.length = value;
  }

  int get length => _innerList.length;

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

  RegularVector pow(double degree, {bool inPlace = false}) {
    List<double> _bufList;

    if (inPlace == true) {
      _bufList = _innerList;
    } else {
      _bufList = new List<double>(this.length);
    }

    for (int i = 0; i < this.length; i++) {
      _bufList[i] = math.pow(this[i], degree);
    }

    if (inPlace == true) {
      return this;
    }

    return new RegularVector.fromList(_bufList);
  }

  double vectorScalarMult(VectorInterface vector) => (this * vector)._sum();

  RegularVector scalarMult(double value, {bool inPlace = false}) {
    List<double> _bufList;

    if (inPlace == true) {
      _bufList = _innerList;
    } else {
      _bufList = new List<double>(this.length);
    }

    for (int i = 0; i < this.length; i++) {
      _bufList[i] = this[i] * value;
    }

    if (inPlace == true) {
      return this;
    }

    return new RegularVector.fromList(_bufList);
  }

  RegularVector scalarAddition(double value, {bool inPlace = false}) {
    List<double> _bufList;

    if (inPlace == true) {
      _bufList = _innerList;
    } else {
      _bufList = new List<double>(this.length);
    }

    for (int i = 0; i < this.length; i++) {
      _bufList[i] = this[i] + value;
    }

    if (inPlace == true) {
      return this;
    }

    return new RegularVector.fromList(_bufList);
  }

  double _sum() => this.reduce((double item, double sum) => item + sum);
}