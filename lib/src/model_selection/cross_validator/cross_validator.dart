import 'package:ml_algo/src/metric/type.dart';
import 'package:ml_algo/src/model_selection/evaluable.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

abstract class CrossValidator<E> {
  double evaluate(Evaluable predictor, MLMatrix<E> points, MLVector<E> labels, MetricType metric,
      {bool isDataNormalized = false});
}
