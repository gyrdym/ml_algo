import 'package:dart_ml/src/math/vector/vector.dart';

abstract class Classifier {
  Vector predictProbabilities(List<Vector> features);
}