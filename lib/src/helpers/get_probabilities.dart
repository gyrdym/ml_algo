import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper.dart';
import 'package:ml_linalg/matrix.dart';

Matrix getProbabilities(Matrix features, Matrix coefficients,
    ScoreToProbMapper scoreToProbMapper) {
  if (features.columnsNum != coefficients.rowsNum) {
    throw Exception('Wrong features number provided: expected '
        '${coefficients.rowsNum}, but ${features.columnsNum} given. '
        'Please, recheck columns number of the passed feature matrix');
  }
  return scoreToProbMapper.map(features * coefficients);
}
