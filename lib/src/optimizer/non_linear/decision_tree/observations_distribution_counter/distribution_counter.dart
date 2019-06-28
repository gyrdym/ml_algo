import 'dart:collection';

abstract class ObservationsDistributionCounter {
  HashMap<T, int> count<T>(Iterable<T> observations);
}
