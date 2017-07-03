import 'package:dart_vector/vector.dart';

abstract class LossFunction {
  double function(Float32x4Vector w, Float32x4Vector x, double y);
}