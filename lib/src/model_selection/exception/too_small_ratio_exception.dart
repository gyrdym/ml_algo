class TooSmallRatioException implements Exception {
  TooSmallRatioException(double ratio, int dataSize)
      : message = 'Ratio is too small comparing to the input data size: ratio '
            '$ratio, min ratio value ${(1 / dataSize).toStringAsFixed(2)}';

  final String message;

  @override
  String toString() => message;
}
