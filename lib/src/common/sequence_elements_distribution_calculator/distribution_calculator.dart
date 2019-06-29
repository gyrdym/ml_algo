import 'dart:collection';

abstract class SequenceElementsDistributionCalculator {
  HashMap<T, double> calculate<T>(Iterable<T> sequence,
      [int classLabelsLength]);
}
