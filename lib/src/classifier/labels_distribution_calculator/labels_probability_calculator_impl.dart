import 'dart:typed_data';

import 'package:ml_algo/src/classifier/labels_distribution_calculator/labels_probability_calculator.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/vector.dart';

class LabelsProbabilityCalculatorImpl implements LabelsProbabilityCalculator {
  final Type dtype;
  final LinkFunction linkFunction;

  LabelsProbabilityCalculatorImpl(this.linkFunction, this.dtype);

  @override
  MLVector getProbabilities(MLVector scores) {
      switch (dtype) {
        case Float32x4:
          return scores.fastMap<Float32x4>((Float32x4 el, int startOffset, int endOffset) => linkFunction(el));
        default:
          throw UnimplementedError();
      }
  }
}
