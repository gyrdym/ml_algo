import 'dart:typed_data' show Float32List;
import 'package:dart_vector/vector.dart';

abstract class Optimizer {
  Float32x4Vector optimize(List<Float32x4Vector> features, Float32List labels, {Float32x4Vector weights});
}