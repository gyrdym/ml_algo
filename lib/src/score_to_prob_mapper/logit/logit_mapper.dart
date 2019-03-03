import 'dart:typed_data';

import 'package:ml_algo/src/score_to_prob_mapper/logit/float32x4_logit_mapper_mixin.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper.dart';
import 'package:ml_linalg/matrix.dart';

class LogitMapper extends Object with Float32x4LogitMapperMixin
    implements ScoreToProbMapper {

  LogitMapper(this.dtype);

  final Type dtype;

  @override
  Matrix linkScoresToProbs(Matrix scores) {
    switch (dtype) {
      case Float32x4:
        return float32x4ScoresToProbs(scores);
      default:
        throw UnsupportedError('Unsupported data type - $dtype');
    }
  }
}
