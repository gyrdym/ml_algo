import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

Matrix addInterceptIf(bool fitIntercept, Matrix observations,
    num interceptScale) =>
  fitIntercept
      ? observations.insertColumns(0, [
        Vector.filled(observations.rowsNum, interceptScale)])
      : observations;
