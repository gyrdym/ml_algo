import 'dart:collection';

import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/iterators/constant.dart';

class ConstantLearningRateIterable with IterableMixin<double> {
  ConstantLearningRateIterable(this._initialValue, this._limit);

  final double _initialValue;
  final int _limit;

  @override
  Iterator<double> get iterator =>
      ConstantLearningRateIterator(_initialValue, _limit);
}
