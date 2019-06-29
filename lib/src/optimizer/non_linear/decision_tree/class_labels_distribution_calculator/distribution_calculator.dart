import 'dart:collection';

abstract class ClassLabelsDistributionCalculator {
  HashMap<T, double> count<T>(Iterable<T> classLabels, int classLabelsLength);
}
