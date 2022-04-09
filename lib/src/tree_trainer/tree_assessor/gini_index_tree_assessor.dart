import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor.dart';
import 'package:ml_linalg/matrix.dart';

class GiniIndexTreeAssessor implements TreeAssessor {
  const GiniIndexTreeAssessor(this.distributionCalculator);

  final DistributionCalculator distributionCalculator;

  @override
  double getAggregatedError(Iterable<Matrix> split, int targetId) {
    var aggregatedGini = 0.0;
    final totalCount =
        split.fold<int>(0, (total, samples) => total + samples.rowsNum);

    for (final samples in split) {
      if (samples.columnsNum == 0) {
        continue;
      }

      if (targetId >= samples.columnsNum) {
        throw ArgumentError.value(
            targetId,
            'targetId',
            'the value should be in [0..${samples.columnsNum - 1}] '
                'range, but given');
      }

      aggregatedGini +=
          getError(samples, targetId) * (samples.rowsNum / totalCount);
    }

    return aggregatedGini;
  }

  @override
  double getError(Matrix samples, int targetId) {
    final target = samples.getColumn(targetId);
    final distribution = distributionCalculator.calculate(target);
    var giniIndex = 0.0;

    distribution.forEach((classLabel, probability) {
      giniIndex += probability * (1 - probability);
    });

    return giniIndex;
  }
}
