import 'dart:math' as math;

import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

part 'softmax_link_function.g.dart';

@JsonSerializable()
class SoftmaxLinkFunction implements LinkFunction {
  const SoftmaxLinkFunction();

  factory SoftmaxLinkFunction.fromJson(Map<String, dynamic> json) =>
      _$SoftmaxLinkFunctionFromJson(json);

  Map<String, dynamic> toJson() => _$SoftmaxLinkFunctionToJson(this);

  @override
  Matrix link(Matrix scores) {
    final maxValues = Vector.fromList(
      scores.rows.map((row) => row.max()).toList(),
      dtype: scores.dtype,
    );

    final rescaledScores = scores.mapColumns((column) => column - maxValues);
    final numerator = rescaledScores.mapElements(math.exp);
    final denominator = numerator
        .reduceColumns((resultColumn, column) => resultColumn + column);

    return numerator.mapColumns((column) => column / denominator);
  }
}
