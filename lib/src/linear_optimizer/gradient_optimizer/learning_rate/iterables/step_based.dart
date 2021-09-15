import 'dart:collection';

import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/iterators/step_based.dart';

class StepBasedLearningRateIterable with IterableMixin<double> {
  StepBasedLearningRateIterable({
    required double initialValue,
    required double decay,
    required int dropRate,
    required int limit,
  })  : _initialValue = initialValue,
        _decay = decay,
        _dropRate = dropRate,
        _limit = limit;

  final double _initialValue;
  final double _decay;
  final int _dropRate;
  final int _limit;

  @override
  Iterator<double> get iterator => StepBasedLearningRateIterator(
        initialValue: _initialValue,
        decay: _decay,
        dropRate: _dropRate,
        limit: _limit,
      );
}
