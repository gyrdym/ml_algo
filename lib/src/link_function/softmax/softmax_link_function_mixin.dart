import 'package:ml_algo/src/link_function/softmax/softmax_link_function.dart';
import 'package:ml_linalg/matrix.dart';

mixin SoftmaxLinkFunctionMixin implements SoftmaxLinkFunction {
  @override
  Matrix link(Matrix scores) {
    final maxValue = scores.max();
    final stableScores = scores - maxValue;
    final numerator = getNumerator(stableScores);
    final normalizerTerm = numerator
        .reduceColumns((resultColumn, scores) => resultColumn + scores);

    return numerator / normalizerTerm;
  }
}
