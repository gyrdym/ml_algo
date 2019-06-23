import 'package:ml_linalg/matrix.dart';
import 'package:xrange/zrange.dart';

abstract class NumberBasedStumpSelector {
  List<Matrix> select(Matrix observations, int selectedColumnIdx,
      ZRange outcomesRange);
}
