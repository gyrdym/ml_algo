class OutRangedRatioException implements Exception {
  OutRangedRatioException(double ratio)
      : message = 'Ratio value must be within the range 0..1 (both exclusive), '
      '$ratio given';

  final String message;

  @override
  String toString() => message;
}
