import 'dart:math';
import 'package:dart_ml/src/math/misc/randomizer/interface/randomizer.dart';

class RandomizerImpl implements Randomizer {
  final Random _generator;

  RandomizerImpl({int seed}) : _generator = new Random(seed);

  int getIntegerFromInterval(int start, int end) => _generator.nextInt(end - start) + start;

  double getDoubleFromInterval(double start, double end) => _generator.nextDouble() * (end - start) + start;
}