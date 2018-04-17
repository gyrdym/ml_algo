import 'dart:math';

import 'package:dart_ml/src/core/math/randomizer/randomizer.dart';

class RandomizerImpl implements Randomizer {
  final Random _generator;

  RandomizerImpl({Random generator, int seed}) : _generator = generator ?? new Random(seed);

  ///returns random interval constrained by [lowerBound] (include) and [upperBound] (exclude)
  List<int> getIntegerInterval(int lowerBound, int upperBound, {int intervalLength = 1}) {
    if (lowerBound == upperBound) {
      throw new RangeError('Lower bound and upper bound must be different');
    }

    List<int> boundaries = _normalizeBoundaries(lowerBound, upperBound);

    if (boundaries.last - boundaries.first < intervalLength) {
      throw new RangeError('Wrong interval given');
    }

    final minPossibleEnd = boundaries.first + intervalLength;
    final end = minPossibleEnd >= boundaries.last ? minPossibleEnd : getIntegerFromInterval(minPossibleEnd, boundaries.last);
    final start = end - intervalLength;

    return [start, end];
  }

  ///returns random interval constrained by [lowerBound] (include) and [upperBound] (exclude)
  List<double> getDoubleInterval(double lowerBound, double upperBound) {
    throw new UnimplementedError('Method isn\'t implemented yet');
  }

  ///returns random integer from interval that starts with [start] (include) and ends with [end] (exclude)
  int getIntegerFromInterval(int start, int end) => _generator.nextInt(end - start) + start;

  ///returns random double from interval that starts with [start] (include) and ends with [end] (exclude)
  double getDoubleFromInterval(double start, double end) {
    if (start == end) {
      throw new RangeError('Start and end values must be different');
    }

    return _generator.nextDouble() * (end - start) + start;
  }

  List<num> _normalizeBoundaries(num start, num end) => start > end ? [end, start] : [start, end];
}