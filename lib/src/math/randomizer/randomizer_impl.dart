import 'dart:math';

import 'package:ml_algo/src/math/randomizer/randomizer.dart';

class RandomizerImpl implements Randomizer {
  RandomizerImpl({Random generator, int seed})
      : _generator = generator ?? Random(seed);

  final Random _generator;

  ///returns random interval constrained by [lowerBound] (include) and [upperBound] (exclude)
  @override
  List<int> getIntegerInterval(int lowerBound, int upperBound,
      {int intervalLength = 1}) {
    if (lowerBound == upperBound) {
      throw RangeError('Lower bound and upper bound must be different');
    }

    final boundaries = _normalizeBoundaries(lowerBound, upperBound);

    if (boundaries.last - boundaries.first < intervalLength) {
      throw RangeError('Wrong interval given');
    }

    final minPossibleEnd = boundaries.first + intervalLength;
    final end = minPossibleEnd >= boundaries.last
        ? minPossibleEnd
        : getIntegerFromInterval(minPossibleEnd, boundaries.last);
    final start = end - intervalLength;

    return [start, end];
  }

  ///returns random interval constrained by [lowerBound] (include) and [upperBound] (exclude)
  @override
  List<double> getDoubleInterval(double lowerBound, double upperBound) {
    throw UnimplementedError('Method isn\'t implemented yet');
  }

  ///returns random integer from interval that starts with [start] (include) and ends with [end] (exclude)
  @override
  int getIntegerFromInterval(int start, int end) =>
      _generator.nextInt(end - start) + start;

  ///returns random double from interval that starts with [start] (include) and ends with [end] (exclude)
  @override
  double getDoubleFromInterval(double start, double end) {
    if (start == end) {
      throw RangeError('Start and end values must be different');
    }

    return _generator.nextDouble() * (end - start) + start;
  }

  List<int> _normalizeBoundaries(int start, int end) =>
      start > end ? [end, start] : [start, end];
}
