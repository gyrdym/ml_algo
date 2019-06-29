import 'dart:collection';

import 'package:ml_algo/src/common/class_labels_distribution_calculator/distribution_calculator.dart';

class ClassLabelsDistributionCalculatorImpl implements
    ClassLabelsDistributionCalculator {
  @override
  HashMap<T, double> calculate<T>(Iterable<T> classLabels,
      [int classLabelsLength]) {
    final length = classLabelsLength ?? classLabels.length;
    final bins = HashMap<T, double>();
    final probabilityStep = 1 / length;
    classLabels.forEach((value) =>
        bins.update(value,
            (existing) => existing + probabilityStep,
            ifAbsent: () => probabilityStep),
    );
    return bins;
  }
}
