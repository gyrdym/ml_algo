class ConstantLearningRateIterator implements Iterator<double> {
  ConstantLearningRateIterator(this._initialValue, this._limit);

  final double _initialValue;
  final int _limit;
  int _iteration = 0;

  @override
  double get current => _initialValue;

  @override
  bool moveNext() => ++_iteration <= _limit;
}
