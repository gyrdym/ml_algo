import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

abstract class DecisionTreeBaseNode {
  DecisionTreeBaseNode(this.splittingValue, this.categoricalValues,
      this.splittingColumnRange);

  final double splittingValue;
  final List<Vector> categoricalValues;
  final ZRange splittingColumnRange;
}
