abstract class Optimizer {
  List<double> errors;
  List<double> optimize(List<List<double>> features, List<double> labels);
}