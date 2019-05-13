import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_type.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper.dart';
import 'package:ml_linalg/dtype.dart';

abstract class ScoreToProbMapperFactory {
  ScoreToProbMapper fromType(ScoreToProbMapperType type, DType dtype);
}
