import 'dart:typed_data';

import 'package:ml_algo/src/classifier/labels_probability_calculator/labels_probability_calculator.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/link_function/link_function_factory_impl.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_linalg/vector.dart';

class LabelsProbabilityCalculatorImpl implements LabelsProbabilityCalculator {
  final Type dtype;
  final LinkFunction linkFunction;

  LabelsProbabilityCalculatorImpl(LinkFunctionType linkFunctionType, this.dtype, {
    LinkFunctionFactory linkFnFactory = const LinkFunctionFactoryImpl(),
  }) : linkFunction = linkFnFactory.fromType(linkFunctionType);

  @override
  MLVector getProbabilities(MLVector scores) {
      switch (dtype) {
        case Float32x4:
          return scores.fastMap<Float32x4>((Float32x4 el, int startOffset, int endOffset) =>
              linkFunction.float32x4Link(el));
        default:
          throw UnsupportedError('Unsupported data type - $dtype');
      }
  }
}
