part of 'package:dart_ml/src/interface.dart';

abstract class Classifier extends Predictor {
  Float32x4Vector predictClasses(List<Float32x4Vector> features);
}