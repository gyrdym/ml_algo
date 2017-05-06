import 'package:dart_ml/src/enums.dart';

abstract class VectorInterface {
  int get length;
  void set length(int value);

  double operator [] (int index);
  void operator []= (int index, double value);

  VectorInterface operator + (VectorInterface vector);
  VectorInterface operator - (VectorInterface vector);
  VectorInterface operator * (VectorInterface vector);
  VectorInterface operator / (VectorInterface vector);

  VectorInterface intPow(int exponent, {bool inPlace = false});
  VectorInterface scalarMult(double value, {bool inPlace = false});
  VectorInterface scalarDivision(double value, {bool inPlace = false});
  VectorInterface scalarAddition(double value, {bool inPlace = false});
  VectorInterface scalarSubtraction(double value, {bool inPlace = false});

  VectorInterface abs({bool inPlace = false});

  double norm([Norm normType = Norm.EUCLIDEAN]);
  double distanceTo(VectorInterface vector, [Norm normType = Norm.EUCLIDEAN]);

  double vectorScalarMult(VectorInterface vector);

  double mean();
  double sum();

  VectorInterface cut(int start, [int end]);
  VectorInterface copy();

  void fill(double value);
  void concat(VectorInterface vector);
}