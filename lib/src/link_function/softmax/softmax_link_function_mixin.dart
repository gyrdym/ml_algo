import 'package:ml_algo/src/link_function/softmax/softmax_link_function.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

mixin SoftmaxLinkFunctionMixin implements SoftmaxLinkFunction {
  @override
  Matrix link(Matrix scores) {
    final maxValues = Vector.fromList(
      scores
        .rows
        .map((row) => row.max())
        .toList(),
      dtype: dtype,
    );

    final rescaledScores = scores.mapColumns((column) => column - maxValues);
    final numerator = getNumerator(rescaledScores);
    final denominator = numerator
        .reduceColumns((resultColumn, column) => resultColumn + column);

    return numerator.mapColumns((column) => column / denominator);
  }
}
