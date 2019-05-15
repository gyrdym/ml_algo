import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

Matrix addInterceptIf(Matrix observations, bool fitIntercept,
    double interceptScale) =>
  fitIntercept
      ? observations.insertColumns(0, [
        Vector.filled(observations.rowsNum, interceptScale)])
      : observations;
