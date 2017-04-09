import 'package:dart_ml/src/enums.dart';

abstract class VectorInterface {
  VectorInterface.fromList(List<double> source);

  double operator [] (int index);
  void operator []= (int index, double value);

  VectorInterface operator + (VectorInterface vector);
  VectorInterface operator - (VectorInterface vector);
  VectorInterface operator * (VectorInterface vector);
  VectorInterface operator / (VectorInterface vector);

  int get length;

  double norm([Norm normType = Norm.EUCLIDEAN]);
  double distanceTo(VectorInterface vector, [Norm normType = Norm.EUCLIDEAN]);

  double vectorScalarMult(VectorInterface vector);

  VectorInterface pow(double degree, {bool inPlace = false});
  VectorInterface scalarMult(double value, {bool inPlace = false});
  VectorInterface scalarAddition(double value, {bool inPlace = false});
}