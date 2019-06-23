import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

abstract class VectorBasedStumpSelector {
  List<Matrix> select(Matrix observations, ZRange range,
      List<Vector> splittingValues);
}
