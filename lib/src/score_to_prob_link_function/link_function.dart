import 'dart:math' as math;

import 'dart:typed_data';

typedef double ScoreToProbLinkFunction(double score);
typedef Float32x4 VectorizedScoreToProbLinkFunction(Float32x4 scores);
typedef Float32x4 VectorizedIndicatorFunction(Float32x4 scores, Float32x4 target);

final zeroes = Float32x4.splat(0.0);
final ones = Float32x4.splat(1.0);

ScoreToProbLinkFunction logitLink =
    (double score) => 1 / (1.0 + math.exp(-score));

VectorizedScoreToProbLinkFunction vectorizedLogitLink =
    (Float32x4 scores) => ones / (ones + (-scores));

VectorizedIndicatorFunction vectorizedIndicator =
    (Float32x4 labels, Float32x4 target) => labels == target ? ones : zeroes;
