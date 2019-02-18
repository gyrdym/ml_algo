import 'dart:typed_data';

import 'package:ml_algo/src/score_to_prob_mapper/float32x4_softmax_mapper_mixin.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class SoftmaxMapper extends Object with Float32x4SoftmaxMapperMixin
    implements ScoreToProbMapper {

  final Type dtype;

  SoftmaxMapper(this.dtype);

  @override
  MLVector linkScoresToProbs(MLVector scores, [MLMatrix scoresByClasses]) {
    switch (dtype) {
      case Float32x4:
        return float32x4ScoresToProbs(scores, scoresByClasses);
      default:
        throw UnsupportedError('Unsupported data type - $dtype');
    }
  }
}