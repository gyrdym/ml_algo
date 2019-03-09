import 'dart:typed_data';

import 'package:ml_algo/src/score_to_prob_mapper/logit/float32x4_inverse_logit_mapper_mixin.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper.dart';
import 'package:ml_linalg/matrix.dart';

class InverseLogitMapper with Float32x4InverseLogitMapper
    implements ScoreToProbMapper {

  InverseLogitMapper(this.dtype);

  final Type dtype;

  @override
  Matrix getProbabilities(Matrix scores) {
    switch (dtype) {
      case Float32x4:
        return float32x4ScoresToProbs(scores);
      default:
        throw UnsupportedError('Unsupported data type - $dtype');
    }
  }
}
