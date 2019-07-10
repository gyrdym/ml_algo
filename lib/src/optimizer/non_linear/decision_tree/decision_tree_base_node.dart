import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

typedef FilterPredicate = bool Function(Vector sample);

abstract class DecisionTreeBaseNode {
  DecisionTreeBaseNode(
      this.isSampleAcceptable,
      this.splittingNumericalValue,
      this.splittingNominalValue,
      this.splittingColumnRange,
  );

  final FilterPredicate isSampleAcceptable;
  final double splittingNumericalValue;
  final Vector splittingNominalValue;
  final ZRange splittingColumnRange;
}
