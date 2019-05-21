import 'package:ml_algo/src/score_to_prob_mapper/logit/float32x4_inverse_logit_mapper_mixin.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

class InverseLogitMapper with Float32x4InverseLogitMapper
    implements ScoreToProbMapper {

  InverseLogitMapper(this.dtype);

  final DType dtype;

  @override
  Matrix map(Matrix scores) {
    switch (dtype) {
      case DType.float32:
        return getFloat32x4Probabilities(scores);
      default:
        throw UnsupportedError('Unsupported data type - $dtype');
    }
  }
}
