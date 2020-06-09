class InvalidClassLabelsException implements Exception {
  InvalidClassLabelsException(num positiveLabel, num negativeLabel) :
        message = 'Positive and negatve labels must be different, both '
            'labels are equal to $positiveLabel instead';

  final String message;

  @override
  String toString() => message;
}
