import 'package:ml_linalg/matrix.dart';

abstract class ObservationsSplitter {
  List<Matrix> split(Matrix observations, int splittingColumnIdx,
      double splittingValue);
}
