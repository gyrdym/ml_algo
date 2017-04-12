import 'dart:collection';
import 'package:dart_ml/src/enums.dart';

abstract class VectorInterface implements ListBase<double> {
  VectorInterface(int dimension);
  VectorInterface.from(List<double> source);
  VectorInterface.filled(int dimension, double value);

  int get dimension;
  void set dimension(int value);

  VectorInterface operator + (VectorInterface vector);
  VectorInterface operator - (VectorInterface vector);
  VectorInterface operator * (VectorInterface vector);
  VectorInterface operator / (VectorInterface vector);

  VectorInterface pow(double degree, {bool inPlace = false});
  VectorInterface scalarMult(double value, {bool inPlace = false});
  VectorInterface scalarDivision(double value, {bool inPlace = false});
  VectorInterface scalarAddition(double value, {bool inPlace = false});
  VectorInterface scalarSubtraction(double value, {bool inPlace = false});

  double norm([Norm normType = Norm.EUCLIDEAN]);
  double distanceTo(VectorInterface vector, [Norm normType = Norm.EUCLIDEAN]);

  double vectorScalarMult(VectorInterface vector);
}