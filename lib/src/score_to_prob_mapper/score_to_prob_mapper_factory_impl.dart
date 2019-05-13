import 'package:ml_algo/src/score_to_prob_mapper/logit/inverse_logit_mapper.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_factory.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_type.dart';
import 'package:ml_algo/src/score_to_prob_mapper/softmax/softmax_mapper.dart';
import 'package:ml_linalg/dtype.dart';

class ScoreToProbMapperFactoryImpl implements ScoreToProbMapperFactory {
  const ScoreToProbMapperFactoryImpl();

  @override
  ScoreToProbMapper fromType(ScoreToProbMapperType type, DType dtype) {
    switch (type) {
      case ScoreToProbMapperType.logit:
        return InverseLogitMapper(dtype);
      case ScoreToProbMapperType.softmax:
        return SoftmaxMapper(dtype);
      default:
        throw UnsupportedError('Unsupported link function type - $type');
    }
  }
}
