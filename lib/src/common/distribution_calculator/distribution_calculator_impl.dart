import 'dart:collection';

import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator.dart';

class DistributionCalculatorImpl implements DistributionCalculator {
  const DistributionCalculatorImpl();

  @override
  HashMap<T, double> calculate<T>(Iterable<T> sequence,
      [int? classLabelsLength]) {
    if (sequence.isEmpty || classLabelsLength == 0) {
      throw Exception('Empty value collection was provided');
    }

    final length = classLabelsLength ?? sequence.length;
    final bins = HashMap<T, double>();
    final probabilityStep = 1 / length;

    sequence.forEach(
      (value) => bins.update(value, (existing) => existing + probabilityStep,
          ifAbsent: () => probabilityStep),
    );

    return bins;
  }
}
