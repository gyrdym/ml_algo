import 'package:ml_linalg/matrix.dart';

abstract class NumericalSplitter {
  List<Matrix> split(Matrix samples, int splittingColumnIdx,
      double splittingValue);

  Iterable<List<int>> getSplittingIndices(Matrix samples,
      int splittingColumnIdx, double splittingValue);
}
