import 'package:ml_algo/src/common/constants/default_parameters/common.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

Matrix addInterceptIf(
        bool fitIntercept, Matrix observations, num interceptScale,
        [DType dtype = dTypeDefaultValue]) =>
    fitIntercept
        ? observations.insertColumns(0, [
            Vector.filled(observations.rowCount, interceptScale, dtype: dtype)
          ])
        : observations;
