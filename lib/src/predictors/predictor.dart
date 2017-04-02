import 'package:dart_ml/src/optimizers/optimizer.dart';

abstract class Predictor {
  void train(List<List<double>> features, List<double> labels);
  List<double> predict(List<List<double>> features);
  List<double> get weights;
  Optimizer get optimizer;
}