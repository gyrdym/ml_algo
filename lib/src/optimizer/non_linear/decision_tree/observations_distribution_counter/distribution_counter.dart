import 'dart:collection';

abstract class ObservationsDistributionCounter {
  HashMap<T, double> count<T>(Iterable<T> values, int valuesLength);
}
