import 'package:ml_linalg/matrix.dart';

abstract class NumberBasedNodeSplitter {
  List<Matrix> split(Matrix observations, int splittingColumnIdx,
      double splittingValue);
}
