class InvalidProbabilityThresholdException implements Exception {
  InvalidProbabilityThresholdException(num value) :
        message = 'Probability threshold should be greater than 0.0 and less '
            'than 1, $value given';

  final String message;

  @override
  String toString() => message;
}
