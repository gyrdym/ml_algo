import 'package:ml_linalg/matrix.dart';

abstract class NodeSplitter {
  List<Matrix> split(Matrix observations, int splittingColumnIdx,
      double splittingValue);
}
