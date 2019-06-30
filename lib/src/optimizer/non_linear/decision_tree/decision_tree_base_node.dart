import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

abstract class DecisionTreeBaseNode {
  DecisionTreeBaseNode(
      this.splittingNumericalValue,
      this.splittingNominalValues,
      this.splittingColumnRange,
  );

  final double splittingNumericalValue;
  final List<Vector> splittingNominalValues;
  final ZRange splittingColumnRange;
}
