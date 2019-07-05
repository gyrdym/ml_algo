import 'package:ml_linalg/matrix.dart';

abstract class SamplesByNumericalValueSplitter {
  List<Matrix> split(Matrix samples, int splittingColumnIdx,
      double splittingValue);
}
