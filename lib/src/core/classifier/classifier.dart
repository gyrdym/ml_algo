import 'package:dart_ml/src/core/predictor/predictor.dart';
import 'package:simd_vector/vector.dart';

abstract class Classifier implements Predictor {
  Float32x4Vector predictClasses(List<Float32x4Vector> features);
}