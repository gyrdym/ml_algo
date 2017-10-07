part of 'package:dart_ml/src/predictor/interface.dart';

abstract class Classifier implements Predictor {
  Float32x4Vector predictClasses(List<Float32x4Vector> features);
}