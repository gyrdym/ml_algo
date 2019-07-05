import 'package:ml_linalg/linalg.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:xrange/zrange.dart';

abstract class SamplesByNominalValueSplitter {
  List<Matrix> split(Matrix samples, ZRange splittingColumnRange,
      List<Vector> nominalValues);
}
