import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/utils/default_parameter_values.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_factory.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_factory_impl.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_type.dart';
import 'package:ml_linalg/linalg.dart';

class LogLikelihoodCost implements CostFunction {
  LogLikelihoodCost(
      ScoreToProbMapperType scoreToProbMapperType, {
        Type dtype = DefaultParameterValues.dtype,
        ScoreToProbMapperFactory scoreToProbMapperFactory =
          const ScoreToProbMapperFactoryImpl(),
      }) : scoreToProbMapper =
  scoreToProbMapperFactory.fromType(scoreToProbMapperType, dtype);

  final ScoreToProbMapper scoreToProbMapper;

  @override
  double getCost(double score, double yOrig) {
    throw UnimplementedError();
  }

  @override
  Matrix getGradient(Matrix x, Matrix w, Matrix y) =>
    x.transpose() * (y - scoreToProbMapper.getProbabilities(x * w));

  @override
  Vector getSubDerivative(int wIdx, Matrix x, Matrix w, Matrix y) =>
      throw UnimplementedError();
}
