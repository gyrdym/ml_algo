import 'dart:collection';

abstract class ClassLabelsDistributionCalculator {
  HashMap<T, double> calculate<T>(Iterable<T> classLabels,
      int classLabelsLength);
}
