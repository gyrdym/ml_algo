import 'dart:collection';

import 'package:ml_algo/src/optimizer/non_linear/decision_tree/observations_distribution_counter/distribution_counter.dart';

class ObservationsDistributionCounterImpl implements
    ObservationsDistributionCounter {
  @override
  HashMap<T, double> count<T>(Iterable<T> values, int valuesLength) {
    final bins = HashMap<T, double>();
    final probabilityStep = 1 / valuesLength;
    values.forEach((value) =>
        bins.update(value,
            (existing) => existing + probabilityStep,
            ifAbsent: () => probabilityStep),
    );
    return bins;
  }
}
