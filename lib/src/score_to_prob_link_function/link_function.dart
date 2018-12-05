typedef double ScoreToProbLinkFunction(double score);
typedef T VectorizedScoreToProbLinkFunction<T>(T scores);
typedef T VectorizedIndicatorFunction<T>(T scores, T target);
