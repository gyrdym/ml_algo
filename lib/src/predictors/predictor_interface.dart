import 'package:dart_ml/src/math/vector_interface.dart';

abstract class PredictorInterface<T extends VectorInterface> {
  void train(List<T> features, List<double> labels);
  T predict(List<T> features);
  T get weights;
}
