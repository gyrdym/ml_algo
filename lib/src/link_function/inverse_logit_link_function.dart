import 'dart:math' as math;

import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/common/exception/logit_scores_matrix_dimension_exception.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/matrix.dart';

part 'inverse_logit_link_function.g.dart';

@JsonSerializable()
class InverseLogitLinkFunction implements LinkFunction {
  static final upperBound = 10;
  static final lowerBound = -10;

  const InverseLogitLinkFunction();

  factory InverseLogitLinkFunction.fromJson(Map<String, dynamic> json) =>
      _$InverseLogitLinkFunctionFromJson(json);

  Map<String, dynamic> toJson() => _$InverseLogitLinkFunctionToJson(this);

  @override
  Matrix link(Matrix scores) {
    if (scores.columnsNum != 1) {
      throw LogitScoresMatrixDimensionException(scores.columnsNum);
    }

    return scores.mapElements(scoreToProbability);
  }

  double scoreToProbability(double score) {
    if (score >= upperBound) {
      return 1;
    }

    if (score <= lowerBound) {
      return 0;
    }

    if (score >= 0) {
      return 1 / (1 + math.exp(-score));
    }

    final exponentToScore = math.exp(score);

    return exponentToScore / (1 + exponentToScore);
  }
}
