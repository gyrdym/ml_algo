import 'package:ml_linalg/matrix.dart';

abstract class SamplesSplitter {
  List<Matrix> split(Matrix samples, int splittingColumnIdx,
      double splittingValue);
}
