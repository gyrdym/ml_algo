import 'package:ml_linalg/matrix.dart';

abstract class SamplesNumericalSplitter {
  List<Matrix> split(Matrix samples, int splittingColumnIdx,
      double splittingValue);
}
