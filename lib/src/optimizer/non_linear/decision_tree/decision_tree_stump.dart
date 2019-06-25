import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class DecisionTreeStump {
  DecisionTreeStump(this.splittingValue, this.categoricalValues,
      this.splittingColumnRange, this.observations);

  final double splittingValue;
  final List<Vector> categoricalValues;
  final ZRange splittingColumnRange;
  final Iterable<Matrix> observations;
}
