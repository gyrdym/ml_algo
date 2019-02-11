import 'dart:typed_data';

import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/default_parameter_values.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_factory.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_factory_impl.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_type.dart';
import 'package:ml_linalg/linalg.dart';

class LogLikelihoodCost implements CostFunction {
  final ScoreToProbMapper scoreToProbMapper;
  final Type dtype;

  LogLikelihoodCost(
    ScoreToProbMapperType scoreToProbMapperType, {
    this.dtype = DefaultParameterValues.dtype,
    ScoreToProbMapperFactory scoreToProbMapperFactory =
        const ScoreToProbMapperFactoryImpl(),
  }) : scoreToProbMapper =
            scoreToProbMapperFactory.fromType(scoreToProbMapperType, dtype);

  @override
  double getCost(double score, double yOrig) {
    throw UnimplementedError();
  }

  @override
  MLVector getGradient(MLMatrix x, MLVector w, MLVector y) {
    final scores = (x * w).toVector();
    switch (dtype) {
      case Float32x4:
        return (x.transpose() *
                (y - scoreToProbMapper.linkScoresToProbs(scores)))
            .toVector();
      default:
        throw throw UnsupportedError('Unsupported data type - $dtype');
    }
  }

  @override
  double getSparseSolutionPartial(int wIdx, MLVector x, MLVector w, double y) =>
      throw UnimplementedError();
}
