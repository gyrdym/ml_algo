import 'dart:collection';

import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator.dart';

class SequenceElementsDistributionCalculatorImpl implements
    SequenceElementsDistributionCalculator {
  const SequenceElementsDistributionCalculatorImpl();

  @override
  HashMap<T, double> calculate<T>(Iterable<T> sequence,
      [int classLabelsLength]) {
    if (sequence.isEmpty || classLabelsLength == 0) {
      throw Exception('Empty value collection is given');
    }
    final length = classLabelsLength ?? sequence.length;
    final bins = HashMap<T, double>();
    final probabilityStep = 1 / length;
    sequence.forEach((value) =>
        bins.update(value,
            (existing) => existing + probabilityStep,
            ifAbsent: () => probabilityStep),
    );
    return bins;
  }
}
