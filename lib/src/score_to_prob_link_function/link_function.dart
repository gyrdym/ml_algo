import 'dart:math' as math;

typedef double ScoreToProbLinkFunction(double score);

ScoreToProbLinkFunction logitLink = (double score) => 1 / (1.0 + math.exp(-score));
