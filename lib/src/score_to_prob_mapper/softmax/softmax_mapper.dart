import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper.dart';
import 'package:ml_algo/src/score_to_prob_mapper/softmax/float32x4_softmax_mapper_mixin.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

class SoftmaxMapper extends Object with Float32x4SoftmaxMapperMixin
    implements ScoreToProbMapper {

  SoftmaxMapper(this.dtype);

  final DType dtype;

  @override
  @override
  Matrix map(Matrix scores) {
    switch (dtype) {
      case DType.float32:
        return float32x4ScoresToProbs(scores);
      default:
        throw UnsupportedError('Unsupported data type - $dtype');
    }
  }
}