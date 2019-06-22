import 'package:ml_linalg/matrix.dart';

abstract class StumpAssessor {
  int getErrorOnStump(Iterable<Matrix> observations);
  int getErrorOnNode(Matrix observations);
}
