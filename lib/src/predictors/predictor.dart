import 'package:dart_ml/src/math/vector_interface.dart';

abstract class Predictor {
  void train(List<VectorInterface> features, VectorInterface labels);
  VectorInterface predict(List<VectorInterface> features);
  VectorInterface get weights;
}