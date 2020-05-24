import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

Matrix addInterceptIf(bool fitIntercept, Matrix observations,
    num interceptScale, [DType dtype = DType.float32]) =>
  fitIntercept
      ? observations.insertColumns(0, [
              Vector
                  .filled(observations.rowsNum, interceptScale, dtype: dtype)])
      : observations;
