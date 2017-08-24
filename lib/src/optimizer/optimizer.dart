part of 'package:dart_ml/src/interface.dart';

abstract class Optimizer {
  Float32x4Vector findMinima(List<Float32x4Vector> features, Float32List labels, {Float32x4Vector weights});
  Float32x4Vector findMaxima(List<Float32x4Vector> features, Float32List labels, {Float32x4Vector weights});
}