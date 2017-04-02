import 'package:dart_ml/src/optimizers/optimizer.dart';

abstract class Predictor {
  void train(List<List<num>> features, List<num> labels);
  List<num> predict(List<List<num>> features);
  List<double> get weights;
  Optimizer get optimizer;
}