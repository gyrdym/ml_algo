import 'dart:collection';

import 'package:ml_algo/src/optimizer/non_linear/decision_tree/observations_distribution_counter/distribution_counter.dart';

class ObservationsDistributionCounterImpl implements
    ObservationsDistributionCounter {
  @override
  HashMap<T, int> count<T>(Iterable<T> observations) {
    final bins = HashMap<T, int>();
    observations.forEach((value) =>
        bins.update(value, (existing) => existing + 1, ifAbsent: () => 1));
    return bins;
  }
}
