import 'dart:collection';

abstract class DistributionCalculator {
  HashMap<T, double> calculate<T>(Iterable<T> sequence,
      [int classLabelsLength]);
}
