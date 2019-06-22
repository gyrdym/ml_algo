import 'package:ml_linalg/matrix.dart';

abstract class StumpAssessor {
  double getErrorOnStump(Iterable<Matrix> observations);
  double getErrorOnNode(Matrix observations);
}
