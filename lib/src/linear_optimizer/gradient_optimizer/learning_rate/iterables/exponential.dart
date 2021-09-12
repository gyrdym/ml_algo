import 'dart:collection';

import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/iterators/exponential.dart';

class ExponentialLearningRateIterable with IterableMixin<double> {
  ExponentialLearningRateIterable({
    required double initialValue,
    required double decay,
    required int limit,
  })  : _initialValue = initialValue,
        _decay = decay,
        _limit = limit;

  final double _initialValue;
  final double _decay;
  final int _limit;

  @override
  Iterator<double> get iterator => ExponentialLearningRateIterator(
        initialValue: _initialValue,
        decay: _decay,
        limit: _limit,
      );
}
