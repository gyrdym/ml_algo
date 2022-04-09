import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor.dart';
import 'package:ml_linalg/matrix.dart';

class GiniIndexTreeSplitAssessor implements TreeSplitAssessor {
  const GiniIndexTreeSplitAssessor(this.distributionCalculator);

  final DistributionCalculator distributionCalculator;

  @override
  double getAggregatedError(Iterable<Matrix> splits, int targetId) {
    var aggregatedGini = 0.0;
    final totalCount =
        splits.fold<int>(0, (total, split) => total + split.rowsNum);

    for (final split in splits) {
      if (split.columnsNum == 0) {
        continue;
      }

      if (targetId >= split.columnsNum) {
        throw ArgumentError.value(
            targetId,
            'targetId',
            'the value should be in [0..${split.columnsNum - 1}] '
                'range, but given');
      }

      aggregatedGini +=
          getError(split, targetId) * (split.rowsNum / totalCount);
    }

    return aggregatedGini / totalCount;
  }

  @override
  double getError(Matrix split, int targetId) {
    final target = split.getColumn(targetId);
    final distribution = distributionCalculator.calculate(target);
    var giniIndex = 0.0;

    distribution.forEach((classLabel, probability) {
      giniIndex += probability * (1 - probability);
    });

    return giniIndex;
  }
}
