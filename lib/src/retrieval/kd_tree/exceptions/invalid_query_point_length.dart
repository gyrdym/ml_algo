class InvalidQueryPointLength implements Exception {
  InvalidQueryPointLength(int pointLength, int expectedLength)
      : message =
            'Invalid query point length: expected length is $expectedLength, but given point\'s length is $pointLength';

  final String message;

  @override
  String toString() => message;
}
