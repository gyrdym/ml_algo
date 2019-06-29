import 'dart:collection';

import 'package:ml_algo/src/optimizer/non_linear/decision_tree/class_labels_distribution_calculator/distribution_calculator.dart';

class ClassLabelsDistributionCalculatorImpl implements
    ClassLabelsDistributionCalculator {
  @override
  HashMap<T, double> count<T>(Iterable<T> classLabels, int classLabelsLength) {
    final bins = HashMap<T, double>();
    final probabilityStep = 1 / classLabelsLength;
    classLabels.forEach((value) =>
        bins.update(value,
            (existing) => existing + probabilityStep,
            ifAbsent: () => probabilityStep),
    );
    return bins;
  }
}
